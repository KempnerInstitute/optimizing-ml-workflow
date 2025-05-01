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
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


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


def setup_distributed():
    """Initialize the distributed process group."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = rank % torch.cuda.device_count()
    return local_rank, rank, world_size


def cleanup_ddp():
    """Destroy the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()

def get_dataloader(batch_size, transform, train_ds="./data/train", val_ds="./data/val", sample_ratio=1.0, num_workers=4, pin_memory=True, distributed=False, rank=0):
    os.makedirs(train_ds, exist_ok=True)
    os.makedirs(val_ds, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load datasets from directories
    train_dataset = torchvision.datasets.ImageFolder(root=train_ds, transform=transform)
    val_dataset = torchvision.datasets.ImageFolder(root=val_ds, transform=transform)

    # Apply sample_ratio to train and validation datasets
    if sample_ratio < 1.0:
        train_size = int(len(train_dataset) * sample_ratio)
        val_size = int(len(val_dataset) * sample_ratio)
        train_dataset = Subset(train_dataset, range(train_size))
        val_dataset = Subset(val_dataset, range(val_size))

    if distributed:
        train_sampler = DistributedSampler(train_dataset, rank=rank)
        val_sampler = DistributedSampler(val_dataset, shuffle=False, rank=rank)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, sampler=val_sampler, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, train_sampler



def build_model(model_name='resnet50', num_classes=1000, finetune=True):
    model = getattr(torchvision.models, model_name)(pretrained=True)
    if finetune:
        for param in model.parameters():
            param.requires_grad = False
    if hasattr(model, 'fc'):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def log_gpu_utilization(csv_writer, wandb_enabled, global_step, rank):
    """Log GPU utilization and memory usage."""
    gpu_utilization = torch.cuda.utilization()
    gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB

    if wandb_enabled and rank == 0:
        wandb.log({"GPU Utilization": gpu_utilization, "GPU Memory (GB)": gpu_memory})

    if csv_writer:
        csv_writer.writerow([global_step, "GPU Utilization", gpu_utilization])
        csv_writer.writerow([global_step, "GPU Memory (GB)", gpu_memory])

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, use_amp, amp_dtype, csv_writer, wandb_enabled, global_step, rank):
    model.train()
    total_loss = 0.0
    loop = tqdm(dataloader, desc=f"Epoch {epoch}", disable=(rank != 0))

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

        # Log to WandB
        if wandb_enabled and rank == 0:
            wandb.log({"Train Loss": loss.item()})

        # Log to CSV
        if csv_writer:
            csv_writer.writerow([global_step, "Train Loss", loss.item()])
            global_step += 1

    return total_loss / len(dataloader), global_step

def validate(model, dataloader, criterion, device, csv_writer, wandb_enabled, epoch, rank):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        loop = tqdm(dataloader, desc="Validation", disable=(rank != 0))
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

    # Log to WandB
    if wandb_enabled and rank == 0:
        wandb.log({"Validation Loss": avg_loss, "Validation Accuracy": accuracy})

    # Log to CSV
    if csv_writer:
        csv_writer.writerow([epoch, "Validation Loss", avg_loss])
        csv_writer.writerow([epoch, "Validation Accuracy", accuracy])

    return avg_loss, accuracy


def training_loop(args, model, train_loader, val_loader, train_sampler, criterion, optimizer, scheduler, scaler, device, csv_writer, use_amp, amp_dtype, rank):
    """Main training loop."""
    early_stopping = EarlyStopping(patience=5)
    best_val_loss = float('inf')
    start_epoch = 1
    global_step = 0

    for epoch in range(start_epoch, args.num_epochs + 1):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        avg_train_loss, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, use_amp, amp_dtype, csv_writer, args.use_wandb, global_step, rank
        )
        avg_val_loss, val_accuracy = validate(
            model, val_loader, criterion, device, csv_writer, args.use_wandb, epoch, rank
        )

        if scheduler:
            scheduler.step()

        if rank == 0:
            logging.info(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Val Accuracy = {val_accuracy:.2f}%")

        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            if rank == 0:
                logging.info("Early stopping triggered.")
            break

        # Log GPU utilization
        log_gpu_utilization(csv_writer, args.use_wandb, global_step, rank)



def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_ds", type=str, required=True, help="Path to the training dataset directory")
    parser.add_argument("--val_ds", type=str, required=True, help="Path to the validation dataset directory")
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
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--scheduler", type=str, default="step", choices=["step", "cosine", "none"])
    parser.add_argument("--mixed_precision", type=str, default="none", choices=["none", "fp16", "bf16", "auto"])
    parser.add_argument("--metrics_csv", type=str, default="metrics.csv")
    parser.add_argument("--tensorboard_csv", type=str, default="tensorboard_logs.csv", help="CSV file for TensorBoard logs")
    parser.add_argument("--pin_memory", type=bool, default=True, help="Whether to use pinned memory for DataLoader")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline"], help="WandB mode: online or offline")
    parser.add_argument("--wandb_project", type=str, default="training-monitoring", help="WandB project name")
    return parser.parse_args()


def main(local_rank, rank, world_size, args):
    logging.basicConfig(filename=args.log_file, level=logging.INFO, format="%(asctime)s - %(message)s")

    device = torch.device("cuda", local_rank)

    # Print system info
    gpu_name = torch.cuda.get_device_name(local_rank)
    capability = torch.cuda.get_device_capability(local_rank)
    print(f"hostname={os.uname()[1]}, global rank={rank}, local_rank={local_rank}, world_size={world_size}, device={device}, gpu_name={gpu_name}, capability={capability}")

    # Mixed precision setup
    supports_bf16 = torch.cuda.is_bf16_supported()
    if args.mixed_precision == "auto":
        args.mixed_precision = "bf16" if supports_bf16 else "fp16"
    if args.mixed_precision == "bf16" and not supports_bf16:
        logging.warning("bf16 not supported on this GPU. Using fp16 instead.")
        args.mixed_precision = "fp16"

    use_amp = args.mixed_precision in ["fp16", "bf16"]
    amp_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Data preparation
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_loader, val_loader, train_sampler = get_dataloader(
        args.batch_size, transform, 
        sample_ratio=args.sample_ratio,
        train_ds=args.train_ds, 
        val_ds=args.val_ds,  
        num_workers=args.num_workers, 
        distributed=True, 
        rank=rank
    )

    # Model, optimizer, and scheduler
    model = build_model(args.model_name, args.num_classes, finetune=args.finetune).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.scheduler == "step":
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    elif args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    else:
        scheduler = None

    # Initialize WandB
    if args.use_wandb and rank == 0:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            group="experiment_group",
            job_type="training"
        )
        wandb.watch(model, log="all", log_freq=100)

    # Open CSV file for TensorBoard logs
    csv_file = open(args.tensorboard_csv, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Step/Epoch", "Metric", "Value"])

    # Training loop
    training_loop(args, model, train_loader, val_loader, train_sampler, criterion, optimizer, scheduler, scaler, device, csv_writer, use_amp, amp_dtype, rank)

    if args.use_wandb and rank == 0:
        wandb.finish()

    csv_file.close()
    cleanup_ddp()


if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()

    # Setup distributed training
    local_rank, rank, world_size = setup_distributed()

    # Call main with distributed parameters
    main(local_rank, rank, world_size, args)


