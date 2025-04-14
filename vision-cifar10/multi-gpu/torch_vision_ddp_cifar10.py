# filename: train_distributed_cifar10.py

import os
import argparse
import logging
import csv
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler, random_split, Subset
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR


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
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = rank % torch.cuda.device_count()
    return local_rank, rank, world_size


def get_dataloader(batch_size, world_size, rank, transform, validation_split=0.2, sample_ratio=1.0, data_path="./data", num_workers=4):
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)

    if sample_ratio < 1.0:
        num_samples = int(len(dataset) * sample_ratio)
        dataset = Subset(dataset, range(num_samples))

    train_size = int((1 - validation_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, train_sampler


def build_model(model_name='resnet50', num_classes=10, finetune=True):
    model_func = getattr(torchvision.models, model_name)
    model = model_func(pretrained=True)
    if finetune:
        for param in model.parameters():
            param.requires_grad = False
    if hasattr(model, 'fc'):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        for param in model.fc.parameters():
            param.requires_grad = True
    return model


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, rank, epoch, use_amp, amp_dtype):
    model.train()
    total_loss = 0.0
    loop = tqdm(dataloader, desc=f"Rank {rank}, Epoch {epoch}", disable=(rank != 0))
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            outputs = model(images)
            loss = criterion(outputs, labels)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device, rank):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        loop = tqdm(dataloader, desc=f"Rank {rank}, Validation", disable=(rank != 0))
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
    amp_mode = checkpoint.get('amp_mode', 'unknown')
    return checkpoint['epoch'], amp_mode


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
    parser.add_argument("--metrics_csv", type=str, default="metrics.csv", help="Path to save CSV metrics")
    args = parser.parse_args()

    logging.basicConfig(filename=args.log_file, level=logging.INFO, format="%(asctime)s - %(message)s")

    local_rank, rank, world_size = setup_distributed()
    device = torch.device("cuda", local_rank)

    gpu_name = torch.cuda.get_device_name(local_rank)
    capability = torch.cuda.get_device_capability(local_rank)
    supports_bf16 = torch.cuda.is_bf16_supported()

    if args.mixed_precision == "auto":
        args.mixed_precision = "bf16" if supports_bf16 else "fp16"

    if args.mixed_precision == "bf16" and not supports_bf16:
        if rank == 0:
            logging.warning("bf16 requested but not supported on this GPU. Training may fail.")

    if not torch.cuda.is_available() and args.mixed_precision != "none":
        if rank == 0:
            logging.warning("AMP requested but no CUDA device found. Training may crash.")

    if rank == 0:
        logging.info(f"Using GPU: {gpu_name}, capability: {capability}, bf16 supported: {supports_bf16}")
        logging.info(f"Using mixed precision: {args.mixed_precision}")

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

    train_loader, val_loader, train_sampler = get_dataloader(
        args.batch_size, world_size, rank, transform,
        sample_ratio=args.sample_ratio, data_path=args.data_path, num_workers=args.num_workers
    )

    model = build_model(args.model_name, num_classes=args.num_classes, finetune=args.finetune).to(device)
    model = DDP(model, device_ids=[local_rank])

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

    if args.resume and os.path.exists("best_model.pth"):
        start_epoch, saved_amp_mode = load_checkpoint(model, optimizer, path="best_model.pth")
        logging.info(f"Resuming training from epoch {start_epoch}, saved AMP mode: {saved_amp_mode}")

    if rank == 0:
        csv_file = open(args.metrics_csv, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy"])

    for epoch in range(start_epoch, args.num_epochs + 1):
        train_sampler.set_epoch(epoch)
        avg_train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, rank, epoch, use_amp, amp_dtype)
        avg_val_loss, val_accuracy = validate(model, val_loader, criterion, device, rank)

        if scheduler:
            scheduler.step()

        if rank == 0:
            mem_allocated = torch.cuda.memory_allocated(device) / (1024**3)
            mem_reserved = torch.cuda.memory_reserved(device) / (1024**3)
            logging.info(f"Memory used: allocated={mem_allocated:.2f}GB, reserved={mem_reserved:.2f}GB")
            logging.info(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Val Accuracy = {val_accuracy:.2f}%")

            csv_writer.writerow([epoch, avg_train_loss, avg_val_loss, val_accuracy])

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_checkpoint(model, optimizer, epoch, path="best_model.pth", amp_mode=args.mixed_precision)
                logging.info(f"Epoch {epoch}: Best model saved with validation loss {avg_val_loss:.4f}")

            early_stopping(avg_val_loss)
            if early_stopping.early_stop:
                logging.info("Early stopping triggered.")
                break

    if rank == 0:
        csv_file.close()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

