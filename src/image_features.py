import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from sklearn.preprocessing import normalize
from PIL import Image
import numpy as np
import io

_model = None
_transform = None

def get_model():
    global _model
    if _model is None:
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model.eval()
        _model = model
    return _model

def get_transform():
    global _transform
    if _transform is None:
        _transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    return _transform

def extract_visual_features(image_bytes):
    """
    Extract visual features from image bytes using ResNet50 and normalize the output.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = get_transform()
    image = transform(image).unsqueeze(0)  # Add batch dimension
    model = get_model()
    with torch.no_grad():
        features = model(image).squeeze().numpy()
    features = normalize(features.reshape(1, -1)).flatten()  # Normalize for better matching
    return features