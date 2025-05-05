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


def get_dataloader(batch_size, transform, validation_split=0.2, sample_ratio=1.0, data_path="./data", num_workers=4, pin_memory=True, distributed=False, rank=0):
    os.makedirs(data_path, exist_ok=True)
    dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)

    if sample_ratio < 1.0:
        num_samples = int(len(dataset) * sample_ratio)
        dataset = Subset(dataset, range(num_samples))

    train_size = int((1 - validation_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    if distributed:
        train_sampler = DistributedSampler(train_dataset, rank=rank)
        val_sampler = DistributedSampler(val_dataset, shuffle=False, rank=rank)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, sampler=val_sampler, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, train_sampler


def build_model(model_name='resnet50', num_classes=10, finetune=True):
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


def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, accuracy, args, rank):
    """Save model checkpoint."""
    if rank != 0:  # Only save on main process
        return
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),  # For DDP models, use .module to get the underlying model
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_accuracy': accuracy
    }
    
    if scheduler:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, args.best_checkpoint_path)
    logging.info(f"Saved best model checkpoint to {args.best_checkpoint_path} with validation loss: {val_loss:.4f} and accuracy: {accuracy:.2f}%")

    if args.use_wandb and rank == 0:
        wandb.log({"Best Validation Loss": val_loss, "Best Validation Accuracy": accuracy})


def export_to_onnx(model, args, input_shape=(1, 3, 224, 224), rank=0):
    """Export model to ONNX format."""
    if rank != 0:  # Only export on main process
        return
    
    if not args.export_onnx:
        return
    
    try:
        # Create a path for the ONNX file
        onnx_path = args.onnx_path if args.onnx_path else os.path.splitext(args.best_checkpoint_path)[0] + '.onnx'
        
        # Load the best checkpoint if available
        if os.path.exists(args.best_checkpoint_path):
            checkpoint = torch.load(args.best_checkpoint_path, map_location='cpu')
            unwrapped_model = model.module  # Get the base model from DDP wrapper
            unwrapped_model.load_state_dict(checkpoint['model_state_dict'])
            logging.info(f"Loaded best model from {args.best_checkpoint_path} for ONNX export")
            print(f"Loaded best model from {args.best_checkpoint_path} for ONNX export")
        else:
            unwrapped_model = model.module  # Get the base model from DDP wrapper
            logging.warning("Best checkpoint not found. Exporting current model state.")
        
        # Set the model to evaluation mode
        unwrapped_model.eval()
        unwrapped_model = unwrapped_model.to('cpu')  # Move model to CPU for export
        
        # Create dummy input tensor for export
        dummy_input = torch.randn(input_shape, requires_grad=True)
        
        # Export the model
        torch.onnx.export(
            unwrapped_model,               # model being run
            dummy_input,                   # model input (or a tuple for multiple inputs)
            onnx_path,                     # where to save the model
            export_params=True,            # store the trained parameter weights inside the model file
            opset_version=12,              # the ONNX version to export the model to
            do_constant_folding=True,      # whether to execute constant folding for optimization
            input_names=['input'],         # the model's input names
            output_names=['output'],       # the model's output names
            dynamic_axes={
                'input': {0: 'batch_size'},    # variable length axes
                'output': {0: 'batch_size'}
            }
        )
        
        logging.info(f"Model successfully exported to ONNX format at: {onnx_path}")
            
    except Exception as e:
        logging.error(f"Error exporting model to ONNX: {e}")


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
    best_val_accuracy = 0.0
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
        
        # Save best model checkpoint
        if args.save_best_checkpoint and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_accuracy = val_accuracy
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, best_val_accuracy, args, rank)
            if rank == 0:
                logging.info(f"New best model saved! Validation loss: {best_val_loss:.4f}, Accuracy: {best_val_accuracy:.2f}%")

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
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--sample_ratio", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--log_file", type=str, default="training_log.txt")
    parser.add_argument("--resume", action="store_true", help="Resume training from a checkpoint or snapshot")
    parser.add_argument("--save_best_checkpoint", action="store_true", help="Save best model checkpoint based on validation loss")
    parser.add_argument("--best_checkpoint_path", type=str, default="best_checkpoint.pth", help="Path to save the best model checkpoint")
    parser.add_argument("--export_onnx", action="store_true", help="Export the best model to ONNX format after training")
    parser.add_argument("--onnx_path", type=str, default="", help="Path to save the ONNX model (default: same as checkpoint with .onnx extension)")
    parser.add_argument("--use_snapshot", action="store_true", help="Enable snapshot functionality")
    parser.add_argument("--snapshot_path", type=str, default="snapshot.pth", help="Path to save/load the snapshot")
    parser.add_argument("--model_name", type=str, default="resnet50")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--num_classes", type=int, default=10)
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
        args.batch_size, transform, sample_ratio=args.sample_ratio, data_path=args.data_path, num_workers=args.num_workers, distributed=True, rank=rank
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
    
    # Export model to ONNX format after training if enabled
    if args.export_onnx:
        print("exporting to onnx model")
        export_to_onnx(model, args, input_shape=(1, 3, 224, 224), rank=rank)

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



