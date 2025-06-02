from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from models.super_resolution import enhance_image
from models.blending import blend_images

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure folders exist
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'enhanced'), exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No file part"
        file = request.files['image']
        if file.filename == '':
            return "No selected file"
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return render_template('home.html', uploaded_image=filepath)
    return render_template('home.html')

@app.route('/enhance', methods=['POST'])
def enhance():
    image_path = request.form.get('image_path')
    if not image_path or not os.path.exists(image_path):
        return "Invalid image path"
    
    enhanced_path = enhance_image(image_path)
    return render_template('home.html', uploaded_image=image_path, enhanced_image=enhanced_path)
@app.route('/blend', methods=['GET', 'POST'])
def blend():
    if request.method == 'POST':
        img1 = request.files.get('image1')
        img2 = request.files.get('image2')
        if not img1 or not img2:
            return "Please upload two images."

        # Save both files
        filename1 = secure_filename(img1.filename)
        filename2 = secure_filename(img2.filename)
        path1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        path2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        img1.save(path1)
        img2.save(path2)

        # Blend images
        blended_path = blend_images(path1, path2)

        return render_template("blend.html", img1=path1, img2=path2, blended=blended_path)
    return render_template("blend.html")

if __name__ == '__main__':
    app.run(debug=True)
