import cv2
import os

MODEL_PATH = os.path.join("models", "espcn_x4.pb")

def enhance_image(image_path):
    # Load the model
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(MODEL_PATH)
    sr.setModel("espcn", 4)

    # Read and enhance image
    image = cv2.imread(image_path)
    result = sr.upsample(image)

    # Save enhanced image
    filename = os.path.basename(image_path)
    enhanced_path = os.path.join("static", "uploads", "enhanced", f"enhanced_{filename}")
    os.makedirs(os.path.dirname(enhanced_path), exist_ok=True)
    cv2.imwrite(enhanced_path, result)

    return enhanced_path
