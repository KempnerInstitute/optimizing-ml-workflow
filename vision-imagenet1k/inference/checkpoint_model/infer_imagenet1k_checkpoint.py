import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import time
import json
import os


def get_model_architecture(model_name, num_classes=1000):
    """
    Create a model with the specified architecture and empty weights.
    Configures the output layer for ImageNet dataset (1000 classes).
    """
    print(f"Creating {model_name} model architecture for {num_classes} classes...")
    
    # Dictionary mapping model names to their initialization functions
    model_builders = {
        'resnet18': lambda: models.resnet18(weights=None),
        'resnet34': lambda: models.resnet34(weights=None),
        'resnet50': lambda: models.resnet50(weights=None),
        'resnet101': lambda: models.resnet101(weights=None),
        'alexnet': lambda: models.alexnet(weights=None),
        'vgg16': lambda: models.vgg16(weights=None),
        'vgg19': lambda: models.vgg19(weights=None),
        'densenet121': lambda: models.densenet121(weights=None),
        'densenet169': lambda: models.densenet169(weights=None),
        'mobilenet_v2': lambda: models.mobilenet_v2(weights=None),
        'mobilenet_v3_small': lambda: models.mobilenet_v3_small(weights=None),
        'mobilenet_v3_large': lambda: models.mobilenet_v3_large(weights=None),
        'efficientnet_b0': lambda: models.efficientnet_b0(weights=None),
        'efficientnet_b1': lambda: models.efficientnet_b1(weights=None),
        'resnext50_32x4d': lambda: models.resnext50_32x4d(weights=None),
    }
    
    if model_name not in model_builders:
        raise ValueError(f"Unsupported model: {model_name}. Supported models: {list(model_builders.keys())}")
    
    # Create model with empty weights
    model = model_builders[model_name]()
    
    # For ImageNet, we don't need to modify the output layer if using standard models
    # as they're already configured for 1000 classes by default
    # But we'll add the code to handle custom class counts
    
    if num_classes != 1000:
        # Modify final layer according to the model type
        if hasattr(model, 'fc'):  # ResNet, DenseNet
            in_features = model.fc.in_features
            model.fc = torch.nn.Linear(in_features, num_classes)
        elif hasattr(model, 'classifier') and isinstance(model.classifier, torch.nn.Sequential):  # VGG, AlexNet
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
        elif hasattr(model, 'classifier') and isinstance(model.classifier, torch.nn.Linear):  # MobileNet
            in_features = model.classifier.in_features
            model.classifier = torch.nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Don't know how to modify the final layer for model: {model_name}")
    
    return model


def load_checkpoint(model, checkpoint_path):
    """
    Load weights from a checkpoint file into the model.
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded model state from checkpoint. Checkpoint info: epoch={checkpoint.get('epoch', 'unknown')}, "
                      f"val_loss={checkpoint.get('val_loss', 'unknown')}, val_accuracy={checkpoint.get('val_accuracy', 'unknown')}")
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                print("Loaded model state_dict from checkpoint.")
            else:
                model.load_state_dict(checkpoint)
                print("Loaded raw state dict from checkpoint.")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded raw state dict from checkpoint.")
        
        return True
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return False


def preprocess_image(image_path, input_size=(224, 224)):
    """
    Preprocess image for inference using the same transforms as during training.
    """
    transform = transforms.Compose([
        transforms.Resize(256),  # Resize to slightly larger than input_size
        transforms.CenterCrop(input_size),  # Standard ImageNet preprocessing
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
        return img_tensor
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None


def load_imagenet_labels(labels_path=None):
    """
    Return the class labels for ImageNet-1K.
    If a labels file is provided, load from there; otherwise, use generic class names.
    """
    if labels_path and os.path.exists(labels_path):
        try:
            with open(labels_path, 'r') as f:
                # Try different formats based on common ImageNet label files
                if labels_path.endswith('.json'):
                    # Handle JSON format like imagenet_class_index.json
                    class_idx = json.load(f)
                    labels = [class_idx[str(i)][1] for i in range(len(class_idx))]
                else:
                    # Handle text file with one class name per line
                    labels = [line.strip() for line in f.readlines()]
                return labels
        except Exception as e:
            print(f"Error loading labels from {labels_path}: {e}")
            print("Using generic class names instead.")
    
    # Return generic class names if no file provided or error occurred
    return [f"class_{i}" for i in range(1000)]


def run_inference(model, image_tensor, device='cpu'):
    """
    Run inference on the preprocessed image tensor.
    """
    model.eval()
    model = model.to(device)
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        start_time = time.time()
        outputs = model(image_tensor)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Get probabilities
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
    return probabilities, inference_time


def main():
    parser = argparse.ArgumentParser(description="Run inference with a PyTorch model on an image for ImageNet-1K classification")
    parser.add_argument("--model", type=str, default="resnet18", help="Model architecture name")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--input_size", type=int, nargs=2, default=[224, 224], help="Input image size (height, width)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on (cpu or cuda)")
    parser.add_argument("--labels", type=str, help="Path to ImageNet labels file")
    parser.add_argument("--num_classes", type=int, default=1000, help="Number of output classes in the model")
    args = parser.parse_args()
    
    # Check if CUDA is available when cuda device is requested
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU")
        args.device = "cpu"
        
    print(f"Using device: {args.device}")
    
    # Create model
    try:
        model = get_model_architecture(args.model, num_classes=args.num_classes)
    except ValueError as e:
        print(e)
        return
        
    # Load checkpoint
    if not load_checkpoint(model, args.checkpoint):
        print("Failed to load checkpoint. Exiting.")
        return
    
    # Preprocess image
    image_tensor = preprocess_image(args.image, tuple(args.input_size))
    if image_tensor is None:
        return
    
    print(f"Input image tensor shape: {image_tensor.shape}")
    
    # Run inference
    print("Running inference...")
    probabilities, inference_time = run_inference(model, image_tensor, device=args.device)
    print(f"Inference completed in {inference_time:.2f} ms")
    
    # Get top predictions
    class_labels = load_imagenet_labels(args.labels)
    top_prob, top_class = torch.topk(probabilities, 5)
    
    # Print results
    print("\nTop 5 predictions:")
    print("-" * 40)
    for i, (prob, class_idx) in enumerate(zip(top_prob, top_class)):
        class_name = class_labels[class_idx.item()]
        print(f"#{i+1}: {class_name} ({prob.item():.4f}, {prob.item()*100:.2f}%)")
    
    # Print raw logits for comparison with ONNX
    print("\nRaw output logits:")
    with torch.no_grad():
        logits = model(image_tensor.to(args.device))[0].cpu().numpy()
    print(f"Logits shape: {logits.shape}")
    print(f"Logits range: min={logits.min():.4f}, max={logits.max():.4f}")
    print("Top 5 logits:")
    top_indices = np.argsort(logits)[::-1][:5]
    for i, idx in enumerate(top_indices):
        print(f"#{i+1}: Class {idx} ({class_labels[idx]}) - {logits[idx]:.4f}")


if __name__ == "__main__":
    main()

