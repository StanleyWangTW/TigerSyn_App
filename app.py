import os
from os.path import join

from flask import Flask, render_template, redirect, url_for
from flask import request, session
import tigersyn
import numpy as np
import matplotlib
from tigersyn.brainage.utils import get_volumes
import tools

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
labels = [
    2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 41, 42, 43, 44, 46, 47, 49,
    50, 51, 52, 53, 54, 58, 60
]


@app.route('/')
def home():
    if 'patient_id' in session:
        return redirect(url_for('upload', patient_id=session['patient_id']))
    else:
        return render_template('cover.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        patient_id = request.form['patient_id']
        session['patient_id'] = patient_id
        return redirect(url_for('upload', patient_id=patient_id))
    else:
        if 'patient_id' in session:
            return redirect(url_for('upload', patient_id=session['patient_id']))
        else:
            return render_template('login.html')


@app.route('/patient=<patient_id>', methods=['GET', 'POST'])
def upload(patient_id):
    if 'patient_id' in session:
        if request.method == 'POST':
            file = request.files['file']
            if '.nii' in file.filename:
                session['img_fname'] = os.path.basename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'image.nii.gz'))
                return redirect(url_for('show', patient_id=session['patient_id']))

        return render_template('upload.html')

    else:
        return redirect(url_for('home'))


@app.route('/logout')
def logout():
    session.pop('patient_id', None)
    return redirect(url_for('home'))


@app.route("/patient=<patient_id>/show", methods=['GET', 'POST'])
def show(patient_id):
    if 'img_fname' in session:

        if request.method == 'POST':
            session['seg_model'] = request.form.get('seg_model')
            return redirect(url_for('segmentation', patient_id=session['patient_id']))

        else:
            raw_img_fname = session['img_fname']
            html2 = tools.view1image(os.path.join(app.config['UPLOAD_FOLDER'], 'image.nii.gz'),
                                     False)

            print('display')
            return render_template('display.html', img_fname=raw_img_fname, image_iframe=html2)

    else:
        return redirect(url_for('home'))


@app.route("/patient=<patient_id>/segmentation", methods=['GET', 'POST'])
def segmentation(patient_id):
    if session['seg_model'] == 'SynthSeg':
        raw_img_fname = session['img_fname']
        img_iframe = tools.view1image(os.path.join(app.config['UPLOAD_FOLDER'], 'image.nii.gz'),
                                      False)
        # segmentation
        tigersyn.run('s', os.path.join(app.config['UPLOAD_FOLDER'], '*.nii.gz'), r'static')
        mask_iframe = tools.view1image(os.path.join('static', 'image_syn.nii.gz'), True)

        # predict brain age
        brain_age = tigersyn.predict_age(os.path.join('static', 'image_syn.nii.gz'))
        brain_age = int(round(brain_age, 0))

        brain_size = get_volumes(os.path.join('static', 'image_syn.nii.gz'), labels)
        brain_size = (np.rint(brain_size)).astype(int)

        vol_data = tools.age_to_json(brain_age, brain_size)

        return render_template('segmentation.html',
                               img_fname=raw_img_fname,
                               image_iframe=img_iframe,
                               mask_iframe=mask_iframe,
                               options=tools.options,
                               vol_data=vol_data)

    if session['seg_model'] == 'Hippocampus':
        # segmentation
        tigersyn.run('h', os.path.join(app.config['UPLOAD_FOLDER'], '*.nii.gz'), r'static')

        # get hippocampus volume
        brain_size = get_volumes(os.path.join('static', 'image_hippocampus.nii.gz'), [1])
        brain_size = (np.rint(brain_size)).astype(int)

        img_fn = session['img_fname']
        img_iframe = tools.view1image(join(app.config['UPLOAD_FOLDER'], 'image.nii.gz'), False)
        mask_iframe = tools.view1image(join('static', 'image_hippocampus.nii.gz'), False)
        hippo_vol = str(brain_size[0])

        return render_template('hippo_seg.html',
                               img_fname=img_fn,
                               image_iframe=img_iframe,
                               mask_iframe=mask_iframe,
                               hippo_vol=hippo_vol)


if __name__ == '__main__':
    app.run(debug=True)
