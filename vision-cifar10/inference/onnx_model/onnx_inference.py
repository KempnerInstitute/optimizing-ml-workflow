import os
import time
import argparse
import numpy as np
import onnxruntime
from PIL import Image


def softmax(x):
    """
    Compute softmax values for each set of scores in x.
    """
    # Subtract max for numerical stability
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def preprocess_image(image_path, input_size=(224, 224)):
    """
    Preprocess the image to match the model input requirements.
    """
    # Open and resize the image
    img = Image.open(image_path).convert('RGB')
    img = img.resize(input_size, Image.LANCZOS)
    
    # Convert to numpy array and normalize - EXPLICITLY USE float32
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    img_array = (img_array - mean) / std
    
    # Transpose from HWC to CHW format (height, width, channels) -> (channels, height, width)
    img_array = img_array.transpose(2, 0, 1)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Double check the data type is float32
    img_array = img_array.astype(np.float32)
    
    print(f"Input data type: {img_array.dtype}")
    return img_array


def load_labels(labels_path):
    """
    Load class labels from a file.
    """
    if not os.path.exists(labels_path):
        # Return default CIFAR-10 labels if file doesn't exist
        return ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                'dog', 'frog', 'horse', 'ship', 'truck']
    
    with open(labels_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels


def run_inference(model_path, image_path, labels_path=None, num_threads=None):
    """
    Run inference with an ONNX model on an image.
    """
    print(f"Loading ONNX model from {model_path}")
    
    # Create session options
    options = onnxruntime.SessionOptions()
    if num_threads is not None:
        options.intra_op_num_threads = num_threads
        options.inter_op_num_threads = num_threads

    # Create ONNX Runtime session
    try:
        session = onnxruntime.InferenceSession(
            model_path, 
            options,
            providers=['CPUExecutionProvider']
        )
    except Exception as e:
        print(f"Error loading the ONNX model: {e}")
        return

    # Get model metadata
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_name = session.get_outputs()[0].name
    
    print(f"Model input name: {input_name}, shape: {input_shape}")
    print(f"Model expected input data type: {session.get_inputs()[0].type}")
    print(f"Model output name: {output_name}")
    
    # Process input image
    # Handle dynamic dimensions in input shape
    if 'batch_size' in str(input_shape):
        input_size = (224, 224)  # Default for most models
    else:
        input_size = (input_shape[2], input_shape[3]) if len(input_shape) == 4 else (224, 224)
    
    print(f"Preprocessing image to size: {input_size}")
    
    try:
        img_data = preprocess_image(image_path, input_size)
        print(f"Image preprocessed, shape: {img_data.shape}")
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return
    
    # Run inference
    print("Running inference...")
    start_time = time.time()
    try:
        outputs = session.run([output_name], {input_name: img_data})
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        print(f"Inference completed in {inference_time:.2f} ms")
    except Exception as e:
        print(f"Error during inference: {e}")
        return
    
    # Process results - apply softmax to convert logits to probabilities
    raw_scores = outputs[0][0]
    print(f"Raw output (logits) range: min={raw_scores.min():.4f}, max={raw_scores.max():.4f}")
    
    # Apply softmax to get probabilities
    probabilities = softmax(raw_scores)
    print(f"After softmax: min={probabilities.min():.4f}, max={probabilities.max():.4f}, sum={probabilities.sum():.4f}")
    
    # Get top 5 predictions
    top_indices = np.argsort(probabilities)[::-1][:5]
    top_scores = probabilities[top_indices]
    
    # Load labels
    labels = load_labels(labels_path) if labels_path else [str(i) for i in range(len(raw_scores))]
    
    # Print results
    print("\nTop 5 predictions:")
    print("-" * 40)
    for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
        if idx < len(labels):
            label = labels[idx]
            print(f"#{i+1}: {label} ({score:.4f}, {score*100:.2f}%)")
        else:
            print(f"#{i+1}: Unknown class {idx} ({score:.4f}, {score*100:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with ONNX model on an image")
    parser.add_argument("--model", required=True, help="Path to ONNX model file")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--labels", help="Path to text file with class labels (one per line)")
    parser.add_argument("--threads", type=int, help="Number of threads to use for inference")
    args = parser.parse_args()
    
    run_inference(args.model, args.image, args.labels, args.threads)

