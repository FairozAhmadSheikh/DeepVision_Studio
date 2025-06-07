from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from models.super_resolution import enhance_image
from models.blending import blend_images
from models.style_transfer import run_style_transfer
from models.denoising import denoise_image
from models.cartoonizer import cartoonize_image
from models.segmenter import segment_image

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
@app.route('/style-transfer', methods=['GET', 'POST'])
def style_transfer():
    if request.method == 'POST':
        content = request.files.get('content')
        style = request.files.get('style')
        if not content or not style:
            return "Please upload both content and style images."

        content_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(content.filename))
        style_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(style.filename))
        content.save(content_path)
        style.save(style_path)

        styled_filename = f"styled_{os.path.basename(content_path)}"
        output_path = os.path.join("static", "uploads", "styled", styled_filename)

        result_path = run_style_transfer(content_path, style_path, output_path)
        return render_template("style_transfer.html", content=content_path, style=style_path, styled=result_path)
    return render_template("style_transfer.html")
@app.route('/denoise', methods=['GET', 'POST'])
def denoise():
    if request.method == 'POST':
        image = request.files.get('image')
        if not image:
            return "Please upload an image."

        filename = secure_filename(image.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(input_path)

        output_path = denoise_image(input_path)
        return render_template("denoise.html", original=input_path, denoised=output_path)

    return render_template("denoise.html")
@app.route('/cartoonize', methods=['GET', 'POST'])
def cartoonize():
    if request.method == 'POST':
        image = request.files.get('image')
        if not image:
            return "Please upload an image."

        filename = secure_filename(image.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(input_path)

        cartoon_path = cartoonize_image(input_path)
        return render_template("cartoonize.html", original=input_path, cartoon=cartoon_path)

    return render_template("cartoonize.html")
@app.route('/segment', methods=['GET', 'POST'])
def segment():
    if request.method == 'POST':
        image = request.files.get('image')
        if not image:
            return "Please upload an image."

        filename = secure_filename(image.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(input_path)

        segmented_path = segment_image(input_path)
        return render_template("segment.html", original=input_path, segmented=segmented_path)

    return render_template("segment.html")
@app.route('/srgan', methods=['GET', 'POST'])
def srgan():
    if request.method == 'POST':
        image = request.files.get('image')
        if not image:
            return "Please upload an image."

        filename = secure_filename(image.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(input_path)

        output_path = enhance_srgan(input_path)
        return render_template("srgan.html", original=input_path, enhanced=output_path)

    return render_template("srgan.html")
if __name__ == '__main__':
    app.run(debug=True)
