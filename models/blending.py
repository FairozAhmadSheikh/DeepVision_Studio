import cv2
import numpy as np
import os

def blend_images(img1_path, img2_path):
    A = cv2.imread(img1_path)
    B = cv2.imread(img2_path)

    # Resize both images to the same size
    height, width = min(A.shape[0], B.shape[0]), min(A.shape[1], B.shape[1])
    A = cv2.resize(A, (width, height))
    B = cv2.resize(B, (width, height))
    # Generate Gaussian pyramids
    G = A.copy()
    gpA = [G]
    for _ in range(6):
        G = cv2.pyrDown(G)
        gpA.append(G)

    G = B.copy()
    gpB = [G]
    for _ in range(6):
        G = cv2.pyrDown(G)
        gpB.append(G)