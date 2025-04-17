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


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


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


def log_system_utilization_to_tensorboard(tensorboard_logdir, epoch, step, cpu_utilization, gpu_utilization, gpu_memory_allocated):
    """
    Logs system utilization metrics (CPU, GPU) to TensorBoard log files.
    """
    if tensorboard_logdir:
        with open(os.path.join(tensorboard_logdir, "system_utilization.txt"), "a") as f:
            f.write(f"{epoch},{step},{cpu_utilization},{gpu_utilization},{gpu_memory_allocated}\n")


def log_system_utilization(device, wandb_enabled=False, tensorboard_logdir=None, epoch=0, step=0):
    # CPU utilization
    cpu_utilization = psutil.cpu_percent(interval=None)

    # GPU utilization (if available)
    gpu_utilization = None
    gpu_memory_allocated = None
    if device.type == "cuda" and torch.cuda.is_available():
        gpu_utilization = torch.cuda.utilization(device)
        gpu_memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)  # Convert to GB

    # Log to wandb
    if wandb_enabled:
        wandb.log({
            "CPU Utilization (%)": cpu_utilization,
            "GPU Utilization (%)": gpu_utilization if gpu_utilization is not None else 0,
            "GPU Memory Allocated (GB)": gpu_memory_allocated if gpu_memory_allocated is not None else 0
        })

    # Log to TensorBoard
    if tensorboard_logdir:
        log_system_utilization_to_tensorboard(
            tensorboard_logdir, epoch, step, cpu_utilization,
            gpu_utilization if gpu_utilization is not None else 0,
            gpu_memory_allocated if gpu_memory_allocated is not None else 0
        )

    return cpu_utilization, gpu_utilization, gpu_memory_allocated


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

        # Log system utilization
        log_system_utilization(
            device, wandb_enabled, tensorboard_logdir,
            epoch=epoch, step=epoch * len(dataloader) + batch_idx
        )

        # Log to TensorBoard and wandb
        if tensorboard_logdir:
            with open(os.path.join(tensorboard_logdir, "train_loss.txt"), "a") as f:
                f.write(f"{epoch * len(dataloader) + batch_idx},{loss.item()}\n")
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
        for batch_idx, (images, labels) in enumerate(loop):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            # Log system utilization
            log_system_utilization(
                device, wandb_enabled, tensorboard_logdir,
                epoch=epoch, step=epoch * len(dataloader) + batch_idx
            )

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    # Log to TensorBoard and wandb
    if tensorboard_logdir:
        with open(os.path.join(tensorboard_logdir, "val_metrics.txt"), "a") as f:
            f.write(f"{epoch},{avg_loss},{accuracy}\n")
    if wandb_enabled:
        wandb.log({"Validation Loss": avg_loss, "Validation Accuracy": accuracy})

    return avg_loss, accuracy



def save_checkpoint(model, optimizer, epoch, path="checkpoint.pth", amp_mode="none"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'amp_mode': amp_mode,
    }, path)


def load_checkpoint(model, optimizer, path="checkpoint.pth"):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint.get('amp_mode', 'none')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--sample_ratio", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--log_file", type=str, default="training_log.txt")
    parser.add_argument("--resume", action="store_true")
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

    early_stopping = EarlyStopping(patience=5)
    best_val_loss = float('inf')
    start_epoch = 1

    # Initialize TensorBoard and wandb
    if args.use_tensorboard:
        os.makedirs(args.tensorboard_logdir, exist_ok=True)

    if args.use_wandb:
        os.environ["WANDB_MODE"] = args.wandb_mode
        wandb.init(project=args.wandb_project, config=vars(args))

    if args.resume and os.path.exists("best_model.pth"):
        start_epoch, _ = load_checkpoint(model, optimizer, path="best_model.pth")
        logging.info(f"Resuming from epoch {start_epoch}")

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

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_checkpoint(model, optimizer, epoch, path="best_model.pth", amp_mode=args.mixed_precision)
                logging.info(f"Epoch {epoch}: Saved best model")

            early_stopping(avg_val_loss)
            if early_stopping.early_stop:
                logging.info("Early stopping triggered.")
                break

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()


