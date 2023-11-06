from flask import Flask, render_template
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

if not os.path.isdir('files'):
    os.mkdir('files')

if not os.path.isdir('static'):
    os.mkdir('static')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'key'
app.config['UPLOAD_FOLDER'] = 'files'
class UploadFileForm(FlaskForm):
    file = FileField('file')
    submit = SubmitField('Upload File')

@app.route("/", methods=['GET', 'POST'])
@app.route("/home", methods=['GET', 'POST'])
def home():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        file.save(os.path.join(
            'files',
            'image.nii.gz'))
    
        img = reorder_img(
            nib.load(os.path.join('files', 'image.nii.gz')),
            resample='nearest').get_fdata()
        
        axial_data = np.rot90(img[:, :, 100])
        cv2.imwrite(r'static\axial.jpg', to_grayscale(axial_data))
        sagittal_data = np.rot90(img[100, :, :])
        cv2.imwrite(r'static\sagittal.jpg', to_grayscale(sagittal_data))
        coronal_data = np.rot90(img[:, 100, :])
        cv2.imwrite(r'static\coronal.jpg', to_grayscale(coronal_data))

    return render_template('index.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)