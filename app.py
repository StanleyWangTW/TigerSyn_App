from flask import Flask, render_template, redirect, request
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename

import nibabel as nib
from nilearn.image import reorder_img
import numpy as np
import cv2

import os

def to_grayscale(img):
    return ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

if not os.path.exists('files'):
    os.makedirs('files')

if not os.path.exists('static'):
    os.makedirs('static')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'files'

@app.route("/", methods=['GET', 'POST'])
@app.route("/home", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename != '':
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'image.nii.gz'))
                
            return redirect('/display')
    
    return render_template('index.html')


@app.route("/display", methods=['GET', 'POST'])
def display():
    img = reorder_img(
        nib.load(os.path.join('files', 'image.nii.gz')),
        resample='nearest').get_fdata()
    
    axial_data = np.rot90(img[:, :, 100])
    cv2.imwrite(r'static\axial.jpg', to_grayscale(axial_data))
    sagittal_data = np.rot90(img[100, :, :])
    cv2.imwrite(r'static\sagittal.jpg', to_grayscale(sagittal_data))
    coronal_data = np.rot90(img[:, 100, :])
    cv2.imwrite(r'static\coronal.jpg', to_grayscale(coronal_data))

    return render_template('display.html')


if __name__ == '__main__':
    app.run(debug=True)