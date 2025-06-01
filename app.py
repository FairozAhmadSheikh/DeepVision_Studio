from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from models.super_resolution import enhance_image

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

if __name__ == '__main__':
    app.run(debug=True)
