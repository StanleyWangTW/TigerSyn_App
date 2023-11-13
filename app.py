import os

from flask import Flask, render_template, redirect, request, session, url_for
import nibabel as nib
from nilearn.image import reorder_img
import tigersyn
import numpy as np
import matplotlib
import cv2

from tools import *

matplotlib.use('agg')
app = Flask(__name__)

if not os.path.isdir('uploads'):
    os.mkdir('uploads')
if not os.path.isdir('static'):
    os.mkdir('static')
if not os.path.isdir('download'):
    os.mkdir('download')

app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'EE304'

# if not os.path.exists('files'):
#     os.makedirs('files')
# if not os.path.exists('static'):
#     os.makedirs('static')

# app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files['file']
        if '.nii' in file.filename:
            session['img_fname'] = os.path.basename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'image.nii.gz'))
            return redirect(url_for('show'))
    
    return render_template('index.html')

@app.route("/show", methods=['GET', 'POST'])
def show():
    if 'img_fname' in session:
        img = reorder_img(
            nib.load(os.path.join(app.config['UPLOAD_FOLDER'], 'image.nii.gz')),
            resample='nearest').get_fdata()

        img_save3GrayScale(img, 'static', 'sagittal.jpg', 'axial.jpg', 'coronal.jpg')

        if request.method == 'POST':
            session['seg_model'] = request.form.get('seg_model')
            print(session['seg_model'])
            return redirect(url_for('segmentation'))
        else:
            return render_template('display.html', raw_img_fname=session['img_fname'])
    else:
        return redirect(url_for('home'))

@app.route("/segmentation", methods=['GET', 'POST'])
def segmentation():
    print(session['seg_model'])
    if session['seg_model'] == 'SynthSeg':
        tigersyn.run('s', os.path.join(app.config['UPLOAD_FOLDER'], '*.nii.gz'), r'static')

        mask = reorder_img(
            nib.load(os.path.join('static', 'image_syn.nii.gz')),
            resample='nearest').get_fdata()
        
        mask_to_3img(mask, 'static', 'gist_ncar')

    # elif session['seg_model'] == 'Hippocampus':
    #     print('hipipipipipipip')
    #     tigersyn.run('h', os.path.join('files', '*.nii.gz'), r'static')
    return render_template('segmentation.html', raw_img_fname=session['img_fname'])


if __name__ == '__main__':
    app.run(debug=True)