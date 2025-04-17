import os
import time
import argparse
import logging
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
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


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, use_amp, amp_dtype):
    model.train()
    total_loss = 0.0
    loop = tqdm(dataloader, desc=f"Epoch {epoch}")


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


def validate(model, dataloader, criterion, device):
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
    total_time = 0.0
    total_memory = 0.0
    total_epoch_passed = 0

    if args.resume and os.path.exists("best_model.pth"):
        start_epoch, _ = load_checkpoint(model, optimizer, path="best_model.pth")
        logging.info(f"Resuming from epoch {start_epoch}")

    with open(args.metrics_csv, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy"])


        for epoch in range(start_epoch, args.num_epochs + 1):
           
            torch.cuda.reset_peak_memory_stats(device)
            start_time = time.time()

            avg_train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, use_amp, amp_dtype)
            avg_val_loss, val_accuracy = validate(model, val_loader, criterion, device)
            


            if scheduler:
                scheduler.step()

            end_time = time.time()
            peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            epoch_time = end_time - start_time
            print(f"[Train] Epoch {epoch} Peak memory: {peak_memory:.2f} GB, Time: {epoch_time:.2f} s")


            logging.info(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Val Accuracy = {val_accuracy:.2f}%")
            csv_writer.writerow([epoch, avg_train_loss, avg_val_loss, val_accuracy])


            total_time += epoch_time
            total_memory += peak_memory
            total_epoch_passed += 1

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_checkpoint(model, optimizer, epoch, path="best_model.pth", amp_mode=args.mixed_precision)
                logging.info(f"Epoch {epoch}: Saved best model")

            early_stopping(avg_val_loss)
            if early_stopping.early_stop:
                logging.info("Early stopping triggered.")
                break
        avg_time = total_time / total_epoch_passed 
        avg_memory = total_memory / total_epoch_passed
        print(f"[Train] Epoch {epoch} - Avg Time: {avg_time:.4f}s, Avg Peak Memory: {avg_memory:.4f} GB")


if __name__ == "__main__":
    main()

