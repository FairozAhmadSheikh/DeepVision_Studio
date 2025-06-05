import cv2
import os

def cartoonize_image(input_path):
    img = cv2.imread(input_path)

    # Resize for consistent output
    img = cv2.resize(img, (512, 512))

    # Step 1: Apply bilateral filter to smooth colors
    color = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    # Step 2: Convert to grayscale and apply median blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 7)
    
    # Step 3: Detect edges
    edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, blockSize=9, C=2)

    # Step 4: Combine edges with the color image
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(color, edges_colored)
