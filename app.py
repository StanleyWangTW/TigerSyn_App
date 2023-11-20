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

# @app.route("/", methods=['GET', 'POST'])
# def home():
#     if request.method == 'POST':
#         file = request.files['file']
#         if '.nii' in file.filename:
#             session['img_fname'] = os.path.basename(file.filename)
#             file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'image.nii.gz'))
#             return redirect(url_for('show'))
    
#     return render_template('index.html')

@app.route("/", methods=['GET', 'POST'])
def home():
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
        #img = reorder_img(
        #    nib.load(os.path.join(app.config['UPLOAD_FOLDER'], 'image.nii.gz')),
        #    resample='nearest').get_fdata()

        #img_save3GrayScale(img, 'static', 'sagittal.jpg', 'axial.jpg', 'coronal.jpg')

        if request.method == 'POST':
            session['seg_model'] = request.form.get('seg_model')
            return redirect(url_for('segmentation', patient_id=session['patient_id']))
        else:
            raw_img_fname = session['img_fname']
            html1 = f'''
{{% extends "upload.html" %}}

{{% block show_image %}}


  <h1 class="text-center">{raw_img_fname}</h1>
  <p></p>
  <h1 class="text-center">MRI Image</h1>
'''
            html2 = view1image(os.path.join(app.config['UPLOAD_FOLDER'], 'image.nii.gz'))
            html3 = '''
  <hr>
  <form method="POST" class="img-thumbnail">
    <h3>選擇執行的模型：</h3>
    <div class="form-check">
      <input class="form-check-input" type="radio" name="seg_model" id="flexRadioDefault1" value="SynthSeg" checked>
      <label class="form-check-label" for="flexRadioDefault1">
        SynthSeg
      </label>
    </div>
    <div class="form-check">
      <input class="form-check-input" type="radio" name="seg_model" id="flexRadioDefault2" value="Hippocampus">
      <label class="form-check-label" for="flexRadioDefault2">
        Hippocampus
      </label>
    </div>
    <div class="col-12">
      <button type="submit" class="btn btn-primary">確定分割</button>
    </div>
  </form>
  {% block show_mask %}{% endblock %}
{% endblock %}            
'''
            return render_template_string(html1 + html2 + html3)
           # return render_template('display.html', raw_img_fname=session['img_fname'])
    else:
        return redirect(url_for('home'))

@app.route("/patient=<patient_id>/segmentation", methods=['GET', 'POST'])
def segmentation(patient_id):
    if session['seg_model'] == 'SynthSeg':
        print("SynthSeg")
        tigersyn.run('s', os.path.join(app.config['UPLOAD_FOLDER'], '*.nii.gz'), r'static')

        #mask = reorder_img(
        #    nib.load(os.path.join('static', 'image_syn.nii.gz')),
        #    resample='nearest').get_fdata()
        
        #mask_to_3img(mask, 'static', 'gist_ncar')

    # elif session['seg_model'] == 'Hippocampus':
    #     print('hipipipipipipip')
    #     tigersyn.run('h', os.path.join('files', '*.nii.gz'), r'static')
        raw_img_fname = session['img_fname']
        html1 = f'''
        {{% extends "upload.html" %}}
    
        {{% block show_image %}}
    
    
          <h1 class="text-center">{raw_img_fname}</h1>
          <p></p>
          <h1 class="text-center">MRI Image</h1>
        '''
        html2 = view1image(os.path.join(app.config['UPLOAD_FOLDER'], 'image.nii.gz'))
        html3 = '''
          <hr>
          <form method="POST" class="img-thumbnail">
            <h3>選擇執行的模型：</h3>
            <div class="form-check">
              <input class="form-check-input" type="radio" name="seg_model" id="flexRadioDefault1" value="SynthSeg" checked>
              <label class="form-check-label" for="flexRadioDefault1">
                SynthSeg
              </label>
            </div>
            <div class="form-check">
              <input class="form-check-input" type="radio" name="seg_model" id="flexRadioDefault2" value="Hippocampus">
              <label class="form-check-label" for="flexRadioDefault2">
                Hippocampus
              </label>
            </div>
            <div class="col-12">
              <button type="submit" class="btn btn-primary">確定分割</button>
            </div>
          </form>
        '''
        html4 = view1image(os.path.join('static', 'image_syn.nii.gz'))
        html5 = '''
          {% block show_mask %}{% endblock %}
        {% endblock %}            
        '''
        return render_template_string(html1 + html2 + html3 + html4 + html5)
    #return render_template('segmentation.html', raw_img_fname=session['img_fname'])
    if session['seg_model'] == 'Hippocampus':
        print("Hippocampus")
        input_data,header = turnDataToInputData(os.path.join(app.config['UPLOAD_FOLDER'], 'image.nii.gz'))
        output = predict("hippo.onnx", input_data, False)
        save_pred = nib.nifti1.Nifti1Image(output.squeeze(), None, header=header)
        nib.save(save_pred, os.path.join('static', "hippo.onnx_image.nii.gz"))
        raw_img_fname = session['img_fname']
        html1 = f'''
                {{% extends "upload.html" %}}

                {{% block show_image %}}


                  <h1 class="text-center">{raw_img_fname}</h1>
                  <p></p>
                  <h1 class="text-center">MRI Image</h1>
                '''
        html2 = view1image(os.path.join(app.config['UPLOAD_FOLDER'], 'image.nii.gz'))
        html3 = '''
                  <hr>
                  <form method="POST" class="img-thumbnail">
                    <h3>選擇執行的模型：</h3>
                    <div class="form-check">
                      <input class="form-check-input" type="radio" name="seg_model" id="flexRadioDefault1" value="SynthSeg" checked>
                      <label class="form-check-label" for="flexRadioDefault1">
                        SynthSeg
                      </label>
                    </div>
                    <div class="form-check">
                      <input class="form-check-input" type="radio" name="seg_model" id="flexRadioDefault2" value="Hippocampus">
                      <label class="form-check-label" for="flexRadioDefault2">
                        Hippocampus
                      </label>
                    </div>
                    <div class="col-12">
                      <button type="submit" class="btn btn-primary">確定分割</button>
                    </div>
                  </form>
                '''
        html4 = view1image(os.path.join('static', "hippo.onnx_image.nii.gz"))
        html5 = '''
                  {% block show_mask %}{% endblock %}
                {% endblock %}            
                '''
        return render_template_string(html1 + html2 + html3 + html4 + html5)

if __name__ == '__main__':
    app.run(debug=True)