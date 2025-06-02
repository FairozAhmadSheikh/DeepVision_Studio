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
    # Generate Laplacian pyramids
    lpA = [gpA[-1]]
    for i in range(5, 0, -1):
        GE = cv2.pyrUp(gpA[i])
        L = cv2.subtract(gpA[i-1], GE)
        lpA.append(L)

    lpB = [gpB[-1]]
    for i in range(5, 0, -1):
        GE = cv2.pyrUp(gpB[i])
        L = cv2.subtract(gpB[i-1], GE)
        lpB.append(L)
    # Add left and right halves
    LS = []
    for la, lb in zip(lpA, lpB):
        cols, rows, ch = la.shape
        ls = np.hstack((la[:, 0:int(cols/2)], lb[:, int(cols/2):]))
        LS.append(ls)