import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")