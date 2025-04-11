import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

def setup_distributed():

    """Initialize distributed training environment and return local rank."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size= dist.get_world_size()
    local_rank = rank % torch.cuda.device_count() 

    print("hostname=", os.uname()[1], "global rank=", rank, "local_rank=", local_rank, "world_size=", world_size)
    print("hostname=", os.uname()[1], "global rank=", dist.get_rank(), "world_size=", dist.get_world_size())
    return local_rank, rank, world_size

def get_dataloader(batch_size, world_size, rank):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return dataloader, sampler

def build_model(num_classes=10, finetune=True):
    model = torchvision.models.resnet50(pretrained=True)
    if finetune:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    for param in model.fc.parameters():
        param.requires_grad = True
    return model

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, rank, epoch):
    model.train()
    total_loss = 0.0
    loop = tqdm(dataloader, desc=f"Rank {rank}, Epoch {epoch}", disable=(rank != 0))
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def main():
    local_rank, rank, world_size = setup_distributed()
    device = torch.device("cuda", local_rank)

    # Configuration
    batch_size = 64
    num_epochs = 20
    learning_rate = 0.001
    finetune = False 

    train_loader, sampler = get_dataloader(batch_size, world_size, rank)
    print("completed loading train data", "rank=", rank)
    model = build_model(num_classes=10, finetune=finetune).to(device)
    print("build model", "rank=", rank)
    model = DDP(model, device_ids=[local_rank])
    print("ddp wrapped model", "rank=", rank)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(1, num_epochs + 1):
        sampler.set_epoch(epoch)
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, rank, epoch)
        if rank == 0:
            print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()

