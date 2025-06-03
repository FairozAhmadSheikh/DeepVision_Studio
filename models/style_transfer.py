import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Preprocessing and deprocessing
loader = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

unloader = transforms.ToPILImage()

def image_loader(image_name):
    image = Image.open(image_name).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)
# Content and Style Loss
class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()

    def forward(self, x):
        self.loss = nn.functional.mse_loss(x, self.target)
        return x

def gram_matrix(x):
    b, c, h, w = x.size()
    features = x.view(b * c, h * w)
    G = torch.mm(features, features.t())
    return G.div(b * c * h * w)
class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, x):
        G = gram_matrix(x)
        self.loss = nn.functional.mse_loss(G, self.target)
        return x

# Build model with inserted loss layers
def get_model_and_losses(cnn, content_img, style_img):
    cnn = cnn.features.to(device).eval()
    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    content_losses = []
    style_losses = []

    model = nn.Sequential()
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            continue
        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_" + name, content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_" + name, style_loss)
            style_losses.append(style_loss)

    return model, style_losses, content_losses
# Run style transfer
def run_style_transfer(content_path, style_path, output_path):
    content_img = image_loader(content_path)
    style_img = image_loader(style_path)
    input_img = content_img.clone()

    cnn = models.vgg19(pretrained=True).to(device).eval()
    model, style_losses, content_losses = get_model_and_losses(cnn, content_img, style_img)

    optimizer = optim.LBFGS([input_img.requires_grad_()])
    
    run = [0]
    while run[0] <= 300:
        def closure():
            optimizer.zero_grad()
            model(input_img)
            style_score = sum(sl.loss for sl in style_losses)
            content_score = sum(cl.loss for cl in content_losses)
            loss = style_score * 1000000 + content_score
            loss.backward()
            run[0] += 1
            return loss
        optimizer.step(closure)

    output_image = input_img.cpu().clone().squeeze(0)
    image = unloader(output_image)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)
    return output_path