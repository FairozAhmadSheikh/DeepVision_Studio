import cv2
import os

def denoise_image(input_path):
    img = cv2.imread(input_path)
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    filename = f"denoised_{os.path.basename(input_path)}"
    output_path = os.path.join("static", "uploads", "denoised", filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, denoised)
    return output_path
