import torch
import torchvision.transforms as T
from torchvision import models
import cv2
from PIL import Image
import os
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained DeepLabV3
model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval().to(device)
# Standard preprocessing
transform = T.Compose([
    T.Resize(520),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])