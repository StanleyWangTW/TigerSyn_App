from flask import Flask, request, render_template
import os
import numpy as np
import torch
import torch.nn
import nibabel as nib
from nilearn.image import reorder_img
import onnxruntime as ort
import cv2
def turnDataToInputData(file_path):
    origin = nib.load(file_path)
    global copy_header
    copy_header = origin.header.copy()
    origin = reorder_img(origin, resample="continuous")
    data = origin.get_fdata()
    data = np.expand_dims(data, axis=0)
    data = np.expand_dims(data, axis=0)
    return data

def predict(model, data, GPU):
    """read array-like data, then segmentation"""
    if GPU and (ort.get_device() == "GPU"):
        ort_sess = ort.InferenceSession(model, providers=['CUDAExecutionProvider'])
    else:
        ort_sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    data_type = 'float32'

    sigmoid = torch.nn.Sigmoid()
    out_sig = sigmoid(torch.tensor(ort_sess.run(None, {'input': data.astype(data_type)})[0]))
    output = out_sig.numpy()
    output[output > 0.5] = 1
    output[output <= 0.5] = 0
    return output

def to_grayscale(img):
    return ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

app = Flask(__name__)
if not os.path.isdir('uploads'):
    os.mkdir('uploads')
if not os.path.isdir('static'):
    os.mkdir('static')
if not os.path.isdir('download'):
    os.mkdir('download')
app.config['UPLOAD_FOLDER'] = 'uploads'
print('dsfdsgdsf')
@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "沒有選擇檔案"
    global file
    file = request.files['file']

    if "nii" in file.filename:
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        return render_template('upload.html',error=f"{file.filename}上傳成功")
    else:
        error="檔案不是nii檔"
        return render_template('upload.html',error=error)

@app.route('/show', methods=['GET'])
def show():
    global input_data
    input_data = turnDataToInputData(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    axial_data = np.rot90(input_data[0,0,:, :, 100])
    cv2.imwrite(r'static\axial.jpg', to_grayscale(axial_data))
    sagittal_data = np.rot90(input_data[0,0,100, :, :])
    cv2.imwrite(r'static\sagittal.jpg', to_grayscale(sagittal_data))
    coronal_data = np.rot90(input_data[0,0,:, 100, :])
    cv2.imwrite(r'static\coronal.jpg', to_grayscale(coronal_data))
    return render_template('preview.html', error=f"{file.filename}上傳成功")

@app.route('/selModel', methods=['POST'])
def selModel():
    global model,output,selModel
    selModel = request.form.get('function')
    if selModel == 'hippocampus':
        model = 'hippo.onnx'
    if selModel == 'Multi_labal':
        model = 'hippo.onnx'
    output = predict(model, input_data, False)
    global x_limit, y_limit, z_limit
    x_limit = output.shape[2] - 1
    y_limit = output.shape[3] - 1
    z_limit = output.shape[4] - 1
    axial_data = np.rot90(output[0, 0, :, :, 100])
    cv2.imwrite(r'static\output_axial.jpg', to_grayscale(axial_data))
    sagittal_data = np.rot90(output[0, 0, 100, :, :])
    cv2.imwrite(r'static\output_sagittal.jpg', to_grayscale(sagittal_data))
    coronal_data = np.rot90(output[0, 0, :, 100, :])
    cv2.imwrite(r'static\output_coronal.jpg', to_grayscale(coronal_data))
    return render_template('output.html', error=f"{file.filename}上傳成功")

@app.route('/slider', methods=['POST'])
def slider():

    data = request.get_json()
    x = int(data['slider1'])
    z = int(data['slider2'])
    y = int(data['slider3'])
    x = int(x * x_limit / 100)
    y = int(y * y_limit / 100)
    z = int(z * z_limit / 100)
    axial_data = np.rot90(input_data[0, 0, :, :, z])
    cv2.imwrite(r'static\axial.jpg', to_grayscale(axial_data))
    sagittal_data = np.rot90(input_data[0, 0, x, :, :])
    cv2.imwrite(r'static\sagittal.jpg', to_grayscale(sagittal_data))
    coronal_data = np.rot90(input_data[0, 0, :, y, :])
    cv2.imwrite(r'static\coronal.jpg', to_grayscale(coronal_data))

    axial_data = np.rot90(output[0, 0, :, :, z])
    cv2.imwrite(r'static\output_axial.jpg', to_grayscale(axial_data))
    sagittal_data = np.rot90(output[0, 0, x, :, :])
    cv2.imwrite(r'static\output_sagittal.jpg', to_grayscale(sagittal_data))
    coronal_data = np.rot90(output[0, 0, :, y, :])
    cv2.imwrite(r'static\output_coronal.jpg', to_grayscale(coronal_data))
    return render_template('output.html', error=f"{file.filename}上傳成功")

@app.route('/download', methods=['GET'])
def download():
    pred_img = nib.nifti1.Nifti1Image(output.squeeze(), None, header=copy_header)
    nib.save(pred_img, os.path.join('download', model+"_"+file.filename))
    return render_template('output.html', error=f"{file.filename}上傳成功")


if __name__ == '__main__':
    app.run()