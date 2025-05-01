import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import requests
from io import BytesIO

# Load a sample image (replace with your own image if needed)
IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg"
response = requests.get(IMAGE_URL)
img = Image.open(BytesIO(response.content)).convert("RGB")

# Define image transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),  # Converts to [0,1] and then normalizes
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Apply transforms
input_tensor = transform(img).unsqueeze(0)  # Add batch dimension

# Load pre-trained ResNet model
model = models.resnet18(pretrained=True)
model.eval()

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
input_tensor = input_tensor.to(device)

# Run inference
with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

# Get top prediction
top1_prob, top1_catid = torch.topk(probabilities, 1)

# Load ImageNet class names
imagenet_labels = requests.get("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt").text.splitlines()

print(f"Predicted class: {imagenet_labels[top1_catid]} ({top1_prob.item():.2%})")


