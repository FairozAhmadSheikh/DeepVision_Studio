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
def segment_image(input_path):
    image = Image.open(input_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()

    # Map class indices to colors (simple colormap)
    colormap = np.random.randint(0, 255, (21, 3), dtype=np.uint8)
    segmented_image = colormap[output_predictions]

    filename = f"segmented_{os.path.basename(input_path)}"
    output_path = os.path.join("static", "uploads", "segmented", filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))

    return output_path