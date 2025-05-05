import argparse
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import time
import json
import os
import requests
from urllib.request import urlopen
from pathlib import Path


def get_pretrained_model(model_name):
    """
    Get a model with pretrained weights from torchvision.
    """
    print(f"Loading pretrained {model_name} model...")
    
    # Dictionary mapping model names to their initialization functions with pretrained weights
    model_builders = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'alexnet': models.alexnet,
        'vgg16': models.vgg16,
        'vgg19': models.vgg19,
        'densenet121': models.densenet121,
        'densenet169': models.densenet169,
        'mobilenet_v2': models.mobilenet_v2,
        'mobilenet_v3_small': models.mobilenet_v3_small,
        'mobilenet_v3_large': models.mobilenet_v3_large,
        'efficientnet_b0': models.efficientnet_b0,
        'efficientnet_b1': models.efficientnet_b1,
        'resnext50_32x4d': models.resnext50_32x4d,
        'inception_v3': models.inception_v3,
        'googlenet': models.googlenet,
        'convnext_tiny': models.convnext_tiny,
        'convnext_small': models.convnext_small,
    }
    
    if model_name not in model_builders:
        raise ValueError(f"Unsupported model: {model_name}. Supported models: {list(model_builders.keys())}")
    
    # Create model with pretrained weights
    model = model_builders[model_name](weights='DEFAULT')
    model.eval()  # Set to evaluation mode
    
    return model


def preprocess_image(image_path, model_name):
    """
    Preprocess image for inference using the appropriate transforms for the model.
    """
    # Special case for Inception models which require 299x299 input
    if model_name in ['inception_v3']:
        input_size = (299, 299)
    else:
        input_size = (224, 224)
    
    transform = transforms.Compose([
        transforms.Resize(int(input_size[0] * 1.14)),  # Resize to slightly larger
        transforms.CenterCrop(input_size),
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


def get_imagenet_labels():
    """
    Download or load the ImageNet class labels.
    """
    # Path to store the labels
    labels_path = Path('imagenet_labels.json')
    
    # If we don't have labels file, download it
    if not labels_path.exists():
        print("Downloading ImageNet labels...")
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        try:
            with urlopen(url) as response:
                labels = response.read().decode('utf-8').splitlines()
                
            # Save to JSON for future use
            with open(labels_path, 'w') as f:
                json.dump(labels, f)
        except Exception as e:
            print(f"Error downloading labels: {e}")
            # Fallback to generic labels
            labels = [f"class_{i}" for i in range(1000)]
    else:
        # Load existing labels
        with open(labels_path, 'r') as f:
            labels = json.load(f)
    
    return labels


def run_inference(model, image_tensor, device='cpu'):
    """
    Run inference on the preprocessed image tensor.
    """
    model = model.to(device)
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        start_time = time.time()
        
        # Handle models with different output formats
        if isinstance(model, models.inception.Inception3) and model.training:
            outputs, _ = model(image_tensor)  # Inception returns (outputs, aux_outputs) in training
        else:
            outputs = model(image_tensor)
            
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Get probabilities
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
    return probabilities, inference_time


def main():
    parser = argparse.ArgumentParser(description="Run inference with pretrained PyTorch models on an image")
    parser.add_argument("--model", type=str, default="resnet50", help="Pretrained model architecture name")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on (cpu or cuda)")
    args = parser.parse_args()
    
    # Check if CUDA is available when cuda device is requested
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU")
        args.device = "cpu"
        
    print(f"Using device: {args.device}")
    
    # Get pretrained model
    try:
        model = get_pretrained_model(args.model)
    except ValueError as e:
        print(e)
        return
    
    # Preprocess image
    image_tensor = preprocess_image(args.image, args.model)
    if image_tensor is None:
        return
    
    print(f"Input image tensor shape: {image_tensor.shape}")
    
    # Run inference
    print("Running inference...")
    probabilities, inference_time = run_inference(model, image_tensor, device=args.device)
    print(f"Inference completed in {inference_time:.2f} ms")
    
    # Get ImageNet labels
    class_labels = get_imagenet_labels()
    
    # Get top predictions
    top_prob, top_class = torch.topk(probabilities, 5)
    
    # Print results
    print("\nTop 5 predictions:")
    print("-" * 40)
    for i, (prob, class_idx) in enumerate(zip(top_prob, top_class)):
        class_name = class_labels[class_idx.item()]
        print(f"#{i+1}: {class_name} ({prob.item():.4f}, {prob.item()*100:.2f}%)")
    

if __name__ == "__main__":
    main()

