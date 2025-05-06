import os
import argparse
import logging
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import wandb


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


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, wandb_enabled=False):
    model.train()
    total_loss = 0.0
    loop = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, (images, labels) in enumerate(loop):
        images, labels = images.to(device, non_blocking=True), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    avg_loss = total_loss / len(dataloader)
    
    if wandb_enabled:
        wandb.log({"Train Loss": avg_loss}, step=epoch)

    return avg_loss


def validate(model, dataloader, criterion, device, wandb_enabled=False, epoch=0):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        loop = tqdm(dataloader, desc="Validation")
        for images, labels in loop:
            images, labels = images.to(device, non_blocking=True), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    if wandb_enabled:
        wandb.log({"Validation Loss": avg_loss, "Validation Accuracy": accuracy}, step=epoch)

    return avg_loss, accuracy


def main():
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    # args = parser.parse_args()
    # TODO: get this working with argparse config input even with sweep
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Initialize wandb
    if config['use_wandb']:
        os.environ["WANDB_MODE"] = config['wandb_mode']
        wandb.init(project=config['wandb_project'], config=config)

    logging.basicConfig(filename=config['log_file'], level=logging.INFO, format="%(asctime)s - %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_loader, val_loader = get_dataloader(
        config['batch_size'], transform,
        sample_ratio=config['sample_ratio'], 
        data_path=config['data_path'], 
        num_workers=config['num_workers']
    )

    model = build_model(config['model_name'], config['num_classes'], finetune=config['finetune']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    if config['scheduler'] == "step":
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    elif config['scheduler'] == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    else:
        scheduler = None

    start_epoch = 1

    for epoch in range(start_epoch, config['num_epochs'] + 1):
        avg_train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, wandb_enabled=config['use_wandb']
        )
        avg_val_loss, val_accuracy = validate(
            model, val_loader, criterion, device, wandb_enabled=config['use_wandb'], epoch=epoch
        )

        if scheduler:
            scheduler.step()

        logging.info(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Val Accuracy = {val_accuracy:.2f}%")
    
    if config['use_wandb']:
        wandb.finish()


if __name__ == "__main__":
    main()

