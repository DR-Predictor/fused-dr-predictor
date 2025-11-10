import torch
from torchvision import models, transforms
from PIL import Image

# Load pretrained ResNet50 once
resnet = models.resnet50(pretrained=True)
resnet.fc = torch.nn.Identity()  # Remove final classification layer
resnet.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet means
        std=[0.229, 0.224, 0.225]    # ImageNet stds
    )
])

def extract_image_features(image: Image.Image) -> torch.Tensor:
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = resnet(img_tensor)
    return features  # Shape: [1, 2048]