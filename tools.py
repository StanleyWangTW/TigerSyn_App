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
    return data, copy_header


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


def get_brain_age_range(age):
    if age <= 20:
        index = 0
    if 20 <= age and age < 30:
        index = 0
    if 30 <= age and age < 40:
        index = 1
    if 40 <= age and age < 50:
        index = 2
    if 50 <= age and age < 60:
        index = 3
    if 60 <= age and age < 70:
        index = 4
    if 70 <= age:
        index = 5
    return index


def get_All_label_brain_sameAgeRange_size():
    All_label_brain_sameAgeRange_size = [
        [244161.06, 249063.96, 248076.72, 243648.15, 229576.55, 240404.2],
        [229271.49, 222264.59, 214510.55, 204848.27, 198633.27, 185512.81],
        [6278.14, 7060.44, 8101.17, 8110.11, 11814.83, 14669.97],
        [171.46, 187.87, 213.17, 239.24, 324.53, 398.39],
        [16067.29, 16464.76, 16988.49, 15706.9, 14700.34, 13894.75],
        [56659.06, 55922.03, 54248.32, 52592.31, 50811.77, 49737.31],
        [8901.46, 8725.05, 8372.03, 7967.38, 7540.21, 7137.82],
        [3510.54, 3391., 3263.31, 3093.66, 3072.81, 3164.31],
        [4808.07, 4726.52, 4550.62, 4329.97, 4147.72, 4200.85],
        [2137.19, 2022.14, 2003.15, 1971.92, 1849.79, 1855.17],
        [818.84, 889.48, 920.72, 1034.69, 1239.41, 1450.99],
        [1598.92, 1602.07, 1529.64, 1584.83, 1628.33, 1639.3],
        [21344.32, 21677.96, 21618.91, 21644.95, 20648.01, 20658.45],
        [4396.81, 4380.25, 4340.38, 4321.66, 4118.9, 3978.08],
        [1541.68, 1564.26, 1513.63, 1509.86, 1410.99, 1290.38],
        [439.89, 409.64, 383.25, 377.71, 342.3, 326.53],
        [4544.9, 4455.56, 4348.54, 4257.33, 4071.67, 4065.84],
        [243408.8, 249038.12, 247329.2, 244231.42, 229243.3, 240037.73],
        [229216.59, 222119.66, 214962.67, 205350.58, 198450.58, 185551.02],
        [5719.7, 6348.05, 7248.81, 7885.5, 11103.18, 13714.88],
        [203.25, 249.26, 270.51, 255.75, 350.13, 370.92],
        [15570.16, 16076.85, 16040.88, 15208.46, 14272.49, 13070.97],
        [57649.64, 57040.42, 55675.64, 53988.32, 51812.98, 50204.61],
        [8067.1, 7850.37, 7635.67, 7373.84, 7030.93, 6855.57],
        [3573.85, 3446.21, 3337.24, 3264.54, 3245.54, 3398.71],
        [4932.24, 4856.11, 4654.08, 4411.09, 4261.32, 4223.53],
        [2166.24, 2104.65, 2054.85, 1961.82, 1861.45, 1840.28],
        [4485.55, 4459.94, 4410.87, 4429.14, 4265.39, 4092.96],
        [1751.15, 1763.6, 1703.03, 1679.74, 1622.41, 1507.57],
        [561.49, 528.74, 505.12, 488.46, 458.1, 444.69],
        [4531.56, 4459.09, 4364.34, 4270.2, 4117.33, 4054.78]
    ]
    return All_label_brain_sameAgeRange_size
