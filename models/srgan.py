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
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)
class UpsampleBlock(nn.Module):
    def __init__(self, channels):
        super(UpsampleBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )

    def forward(self, x):
        return self.block(x)
# Load pre-trained generator
def load_srgan_model():
    model = Generator().to(device)
    model.load_state_dict(torch.load("models/weights/SRGAN_Generator.pth", map_location=device))
    model.eval()
    return model

# Enhance image
def enhance_srgan(input_path):
    model = load_srgan_model()

    image = Image.open(input_path).convert("RGB")
    lr = ToTensor()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        sr = model(lr).squeeze(0).cpu()
    sr_image = ToPILImage()(sr.clamp(0.0, 1.0))

    filename = f"srgan_{os.path.basename(input_path)}"
    output_path = os.path.join("static", "uploads", "srgan", filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sr_image.save(output_path)

    return output_path