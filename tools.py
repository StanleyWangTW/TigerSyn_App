import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
from nilearn import plotting, image
from flask import render_template_string
from nilearn.image import reorder_img
import onnxruntime as ort
import torch
import torch.nn
import nibabel as nib
def to_grayscale(img):
    return ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)


def img_save3GrayScale(img, dir, saggital='saggital.jpg', axial='axial.jpg', coronal='coronal.jpg'):
    sagittal_data = np.rot90(img[100, :, :])
    cv2.imwrite(os.path.join(dir, saggital), to_grayscale(sagittal_data))

    axial_data = np.rot90(img[:, :, 100])
    cv2.imwrite(os.path.join(dir, axial), to_grayscale(axial_data))

    coronal_data = np.rot90(img[:, 100, :])
    cv2.imwrite(os.path.join(dir, coronal), to_grayscale(coronal_data))


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

def view1image(path1):
    nii_file1 = path1
    img1 = image.load_img(nii_file1)
    view = plotting.view_img(img1, black_bg=True, colorbar=False, bg_img=False)
    html_content = view.get_iframe()
    html = f'<br><div style="display: flex; justify-content: center;"><div><html><body>{html_content}</body></html>'
    return render_template_string(html)

def view2image(path1, path2):
    nii_file1 = path1  # 更換為您檔案的路徑
    nii_file2 = path2
    img1 = image.load_img(nii_file1)
    img2 = image.load_img(nii_file2)
    view = plotting.view_img(img1, black_bg=True, colorbar=False, bg_img=False)
    view2 = plotting.view_img(img2, black_bg=True, colorbar=False, bg_img=False)
    html_content = view.get_iframe()
    html_content2 = view2.get_iframe()
    html = f'<br><html><body>{html_content}<br>{html_content2}</body></html>'
    return render_template_string(html)

def turnDataToInputData(file_path):
    origin = nib.load(file_path)
    copy_header = origin.header.copy()
    origin = reorder_img(origin, resample="continuous")
    data = origin.get_fdata()
    data = np.expand_dims(data, axis=0)
    data = np.expand_dims(data, axis=0)
    return data,copy_header

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