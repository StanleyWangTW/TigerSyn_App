from flask import Flask, render_template, redirect, request, session, url_for

import nibabel as nib
from nilearn.image import reorder_img
import tigersyn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2

import glob
import time
import os

matplotlib.use('agg')

def to_grayscale(img):
    return ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

def mask_to_3img(mask, dir, cmap):
    plt.figure()
    plt.imshow(np.rot90(mask[100, :, :]), cmap=cmap)
    plt.axis('off')
    plt.savefig(os.path.join(dir, 'sagittal_mask.jpg'), bbox_inches='tight')

    plt.figure()
    plt.imshow(np.rot90(mask[:, :, 100]), cmap=cmap)
    plt.axis('off')
    plt.savefig(os.path.join(dir, 'axial_mask.jpg'), bbox_inches='tight')

    plt.figure()
    plt.imshow(np.rot90(mask[:, 100, :]), cmap=cmap)
    plt.axis('off')
    plt.savefig(os.path.join(dir, 'coronal_mask.jpg'), bbox_inches='tight')


if not os.path.exists('files'):
    os.makedirs('files')

if not os.path.exists('static'):
    os.makedirs('static')

app = Flask(__name__)
app.secret_key = 'EE304'
app.config['UPLOAD_FOLDER'] = 'files'

@app.route("/", methods=['GET', 'POST'])
@app.route("/home", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename != '':
            session['img_fname'] = os.path.basename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'image.nii.gz'))
            return redirect('/display')
    else:
        return render_template('index.html')


@app.route("/display", methods=['GET', 'POST'])
def display():
    if 'img_fname' in session:
        img = reorder_img(
            nib.load(os.path.join('files', 'image.nii.gz')),
            resample='nearest').get_fdata()
        
        session['v1'] = int(request.form.get('sag')) if request.form.get('sag') is not None else 0
        print(session['v1'])

        sagittal_data = np.rot90(img[100, :, :])
        cv2.imwrite(r'static\sagittal.jpg', to_grayscale(sagittal_data))

        axial_data = np.rot90(img[:, :, 100])
        cv2.imwrite(r'static\axial.jpg', to_grayscale(axial_data))

        coronal_data = np.rot90(img[:, 100, :])
        cv2.imwrite(r'static\coronal.jpg', to_grayscale(coronal_data))

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
        tigersyn.run('s', os.path.join('files', '*.nii.gz'), r'static')

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