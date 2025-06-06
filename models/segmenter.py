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