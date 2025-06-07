import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Generator model (from original SRGAN paper)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU(),
            *[ResidualBlock(64) for _ in range(5)],
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            UpsampleBlock(64),
            UpsampleBlock(64),
            nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)
