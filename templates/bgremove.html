import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from models.u2net_arch import U2NET  # Download this file separately (model architecture)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_u2net_model():
    model = U2NET(3, 1)
    model.load_state_dict(torch.load("models/weights/u2net.pth", map_location=device))
    model.to(device)
    model.eval()
    return model


    def remove_background(image_path):
    model = load_u2net_model()

    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        d1, *_ = model(img_tensor)
        pred = d1[:, 0, :, :]
        pred = (pred - pred.min()) / (pred.max() - pred.min())
        mask = pred.squeeze().cpu().numpy()

    mask = Image.fromarray((mask * 255).astype(np.uint8)).resize(image.size, Image.LANCZOS)
    empty = Image.new("RGBA", image.size)
    image.putalpha(mask)

    output_path = os.path.join("static", "uploads", "bgremove", f"bgremoved_{os.path.basename(image_path).split('.')[0]}.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path, format='PNG')

    return output_path
