# 🧠 DeepVision Studio — AI-Powered Image Processing Web App

**DeepVision Studio** is a full-featured Flask web application that enables advanced image processing using deep learning. Built with scalability in mind, the app allows users to enhance, transform, and blend images through a clean web interface, using state-of-the-art computer vision models.

This project is being developed incrementally over multiple days — a new AI/ML feature is added each day, making it ideal for learning, showcasing, and real-world applications.

---

## 🚀 Features

- 📷 **Image Upload & Preview**  
  Upload and preview images in the browser with automatic handling.

- 🔍 **Super-Resolution Enhancement**  
  Uses OpenCV's DNN Super Resolution module with the **ESPCN** model to upscale and sharpen low-res images.

- 🔧 **Modular Design**  
  Clean codebase ready to integrate features like:
  - Image blending (Laplacian Pyramid, Seamless Cloning)
  - Noise reduction
  - Style transfer
  - Object detection
  - Segmentation

- 💡 **Deep Learning Ready**  
  Uses pre-trained models for high-quality results without requiring GPU setup.

---

## 📁 Tech Stack

- **Backend**: Python, Flask  
- **Image Processing**: OpenCV (`cv2.dnn_superres`)  
- **Frontend**: HTML, CSS (Jinja2 templating)  
- **Model**: ESPCN (Efficient Sub-Pixel Convolutional Neural Network)  
- **Deployment Ready**: Lightweight & extensible for local or cloud deployment

---



---

## 🛠️ Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/FairozAhmadSheikh/DeepVision_Studio
cd DeepVision_Studio
```

### 2.Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Run the App
```bash
python app.py
```
Then open your browser and visit:
http://localhost:5000

# 💡 Want to contribute or suggest a feature?
Open an issue or submit a pull request!

###  📌 Project Timeline (Development Log)

# | Day | Feature                                 | Status      |
| --- | --------------------------------------- | ----------- |
| 1   | Basic Image Upload & Display            | ✅ Done      |
| 2   | Super-Resolution Enhancement (ESPCN)    | ✅ Done      |
| 3   | Image Blending using OpenCV             | ✅ Done      |
| 4   | Style Transfer with VGG (Neural Style)  | ✅ Done      |
| 5   | Image Denoising (fastNlMeansDenoising)  | ✅ Done      |
| 6   | Cartoonize Image using Bilateral Filter | ✅ Done      |
| 7   | Semantic Segmentation (DeepLabV3)       | ✅ Done      |
| 8   | Super-Resolution (SRGAN or Upscaling)   | 🔜 Upcoming |
| 9   | Background Removal (U^2-Net / Deeplab)  | 🔜 Upcoming |
| 10  | Object Detection (YOLOv5 or SSD)        | 🔜 Upcoming |
| 11  | Image Colorization (Grayscale to Color) | 🔜 Upcoming |
| 12  | Face Detection & Blur                   | 🔜 Upcoming |
| 13  | Image Caption Generator (CNN + RNN)     | 🔜 Upcoming |
| 14  | Text Detection (EAST or OCR)            | 🔜 Upcoming |
| 15  | Depth Estimation                        | 🔜 Upcoming |
| 16  | Real-Time Style Transfer (Webcam-based) | 🔜 Upcoming |
| 17  | Face Swapping                           | 🔜 Upcoming |
| 18  | AnimeGAN Filter                         | 🔜 Upcoming |
| 19  | Full PDF Export of All Results          | 🔜 Upcoming |
| 20  | Final UI Polishing & Dockerization      | 🔜 Upcoming |



### 📜 License
This project is licensed under the MIT License.
Feel free to fork, use, or contribute to it.

