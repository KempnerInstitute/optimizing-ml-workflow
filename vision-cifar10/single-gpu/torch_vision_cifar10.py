import os
import time
import argparse
import logging
import csv
import psutil  # For CPU utilization
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import wandb

def early_stopping(val_loss, best_loss, patience_counter, patience=5, delta=0):
    """
    Returns:
        tuple: (should_stop, updated_best_loss, updated_patience_counter)
            - should_stop (bool): Whether to stop training.
            - updated_best_loss (float): Updated best validation loss.
            - updated_patience_counter (int): Updated patience counter.
    """
    if val_loss < best_loss - delta:
        best_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            return True, best_loss, patience_counter

    return False, best_loss, patience_counter


def get_dataloader(batch_size, transform, validation_split=0.2, sample_ratio=1.0, data_path="./data", num_workers=4, pin_memory=True):
    os.makedirs(data_path, exist_ok=True)
    dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)

    if sample_ratio < 1.0:
        num_samples = int(len(dataset) * sample_ratio)
        dataset = Subset(dataset, range(num_samples))

    train_size = int((1 - validation_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader


def build_model(model_name='resnet50', num_classes=10, finetune=True):
    model = getattr(torchvision.models, model_name)(pretrained=True)
    if finetune:
        for param in model.parameters():
            param.requires_grad = False
    if hasattr(model, 'fc'):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def save_checkpoint(model, optimizer, epoch, path="checkpoint.pth", amp_mode="none"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'amp_mode': amp_mode,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path="checkpoint.pth"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    amp_mode = checkpoint.get('amp_mode', 'none')
    print(f"Checkpoint loaded from {path}, resuming from epoch {epoch}")
    return epoch, amp_mode


def save_snapshot(model, optimizer, scheduler, scaler, epoch, best_val_loss, snapshot_path="snapshot.pth"):
    snapshot = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'best_val_loss': best_val_loss
    }
    torch.save(snapshot, snapshot_path)
    print(f"Snapshot saved to {snapshot_path}")


def load_snapshot(model, optimizer, scheduler, scaler, snapshot_path="snapshot.pth"):
    if not os.path.exists(snapshot_path):
        raise FileNotFoundError(f"Snapshot file not found: {snapshot_path}")

    snapshot = torch.load(snapshot_path)
    model.load_state_dict(snapshot['model_state_dict'])
    optimizer.load_state_dict(snapshot['optimizer_state_dict'])
    if scheduler and snapshot['scheduler_state_dict']:
        scheduler.load_state_dict(snapshot['scheduler_state_dict'])
    if scaler and snapshot['scaler_state_dict']:
        scaler.load_state_dict(snapshot['scaler_state_dict'])
    epoch = snapshot['epoch']
    best_val_loss = snapshot['best_val_loss']
    print(f"Snapshot loaded from {snapshot_path}, resuming from epoch {epoch}")
    return epoch, best_val_loss


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, use_amp, amp_dtype, tensorboard_logdir=None, wandb_enabled=False):
    model.train()
    total_loss = 0.0
    loop = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, (images, labels) in enumerate(loop):
        images, labels = images.to(device), labels.to(device)
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            outputs = model(images)
            loss = criterion(outputs, labels)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

        # Log to wandb
        if wandb_enabled:
            wandb.log({"Train Loss": loss.item()})

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device, tensorboard_logdir=None, wandb_enabled=False, epoch=0):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        loop = tqdm(dataloader, desc="Validation")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    # Log to wandb
    if wandb_enabled:
        wandb.log({"Validation Loss": avg_loss, "Validation Accuracy": accuracy})

    return avg_loss, accuracy


def arg_parser():
    """
    Creates and returns the argument parser for the program.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--sample_ratio", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--log_file", type=str, default="training_log.txt")
    parser.add_argument("--resume", action="store_true", help="Resume training from a checkpoint or snapshot")
    parser.add_argument("--use_checkpoint", action="store_true", help="Enable checkpoint saving/loading")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint.pth", help="Path to save/load the checkpoint")
    parser.add_argument("--use_snapshot", action="store_true", help="Enable snapshot functionality")
    parser.add_argument("--snapshot_path", type=str, default="snapshot.pth", help="Path to save/load the snapshot")
    parser.add_argument("--model_name", type=str, default="resnet50")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--scheduler", type=str, default="step", choices=["step", "cosine", "none"])
    parser.add_argument("--mixed_precision", type=str, default="none", choices=["none", "fp16", "bf16", "auto"])
    parser.add_argument("--metrics_csv", type=str, default="metrics.csv")
    parser.add_argument("--pin_memory", type=bool, default=True, help="Whether to use pinned memory for DataLoader")
    parser.add_argument("--use_tensorboard", action="store_true", help="Enable TensorBoard logging")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline"], help="WandB mode: online or offline")
    parser.add_argument("--wandb_project", type=str, default="training-monitoring", help="WandB project name")
    parser.add_argument("--tensorboard_logdir", type=str, default="./tensorboard_logs", help="Directory for TensorBoard logs")
    return parser


def main():
    parser = arg_parser()
    args = parser.parse_args()

    logging.basicConfig(filename=args.log_file, level=logging.INFO, format="%(asctime)s - %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    supports_bf16 = torch.cuda.is_bf16_supported()
    if args.mixed_precision == "auto":
        args.mixed_precision = "bf16" if supports_bf16 else "fp16"
    if args.mixed_precision == "bf16" and not supports_bf16:
        logging.warning("bf16 not supported on this GPU. Using fp16 instead.")
        args.mixed_precision = "fp16"

    use_amp = args.mixed_precision in ["fp16", "bf16"]
    amp_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_loader, val_loader = get_dataloader(
        args.batch_size, transform,
        sample_ratio=args.sample_ratio, data_path=args.data_path, num_workers=args.num_workers
    )

    model = build_model(args.model_name, args.num_classes, finetune=args.finetune).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.scheduler == "step":
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    elif args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    else:
        scheduler = None

    # Initialize variables for early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5  # Number of epochs to wait before stopping
    delta = 0  # Minimum improvement threshold

    start_epoch = 1

    # Resume from checkpoint if specified and enabled
    if args.use_checkpoint and args.resume and os.path.exists(args.checkpoint_path):
        start_epoch, _ = load_checkpoint(model, optimizer, path=args.checkpoint_path)

    # Resume from snapshot if specified and enabled
    if args.use_snapshot and args.resume and os.path.exists(args.snapshot_path):
        start_epoch, best_val_loss = load_snapshot(model, optimizer, scheduler, scaler, args.snapshot_path)

    # Initialize wandb
    if args.use_wandb:
        os.environ["WANDB_MODE"] = args.wandb_mode
        wandb.init(project=args.wandb_project, config=vars(args))

    # Initialize TensorBoard log directory
    if args.use_tensorboard:
        os.makedirs(args.tensorboard_logdir, exist_ok=True)



    with open(args.metrics_csv, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy"])

        for epoch in range(start_epoch, args.num_epochs + 1):
            avg_train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, device, epoch, use_amp, amp_dtype,
                tensorboard_logdir=args.tensorboard_logdir if args.use_tensorboard else None, wandb_enabled=args.use_wandb
            )
            avg_val_loss, val_accuracy = validate(
                model, val_loader, criterion, device,
                tensorboard_logdir=args.tensorboard_logdir if args.use_tensorboard else None, wandb_enabled=args.use_wandb, epoch=epoch
            )

            if scheduler:
                scheduler.step()

            logging.info(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Val Accuracy = {val_accuracy:.2f}%")
            csv_writer.writerow([epoch, avg_train_loss, avg_val_loss, val_accuracy])

            # Save checkpoint if enabled
            if args.use_checkpoint:
                save_checkpoint(model, optimizer, epoch, path=args.checkpoint_path, amp_mode=args.mixed_precision)

            # Save snapshot if enabled
            if args.use_snapshot:
                save_snapshot(model, optimizer, scheduler, scaler, epoch, best_val_loss, args.snapshot_path)


            # Check for early stopping
            should_stop, best_val_loss, patience_counter = early_stopping(
                avg_val_loss, best_val_loss, patience_counter, patience=patience, delta=delta
            )
            if should_stop:
                logging.info("Early stopping triggered.")
                break

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()


