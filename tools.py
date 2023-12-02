import os
import numpy as np
import cv2
import  matplotlib as mpl
import matplotlib.pyplot as plt
from nilearn import plotting, image
from flask import render_template_string
from nilearn.image import reorder_img
import onnxruntime as ort
import torch
import torch.nn
import nibabel as nib
from matplotlib.colors import LinearSegmentedColormap
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps
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

def view1image(path1,colormap):
    nii_file1 = path1
    img1 = image.load_img(nii_file1)
    if colormap:
        view = plotting.view_img(img1, black_bg=True, colorbar=False, bg_img=False,cmap=get_cmap(labels))
    else:
        view = plotting.view_img(img1, black_bg=True, colorbar=False, bg_img=False ,cmap=nilearn_cmaps["brown_blue"])#,cmap=plt.cm.get_cmap('gray')
    html_content = view.get_iframe()
    html = f'<body><div style="display: flex; justify-content: center;">{html_content}</div></body>'
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

def get_brain_age_range(age):
    if age<=20:
        index = 0
    if 20<=age and age<30:
        index = 0
    if 30<=age and age<40:
        index = 1
    if 40<=age and age<50:
        index = 2
    if 50<=age and age<60:
        index = 3
    if 60<=age and age<70:
        index = 4
    if 70<=age:
        index = 5
    return index

def get_diagram_point(vol , index , vol_index):
    if index == 0:
        list =  [vol[vol_index], 0, 0, 0, 0, 0]
    if index == 1:
        list =  [0, vol[vol_index], 0, 0, 0, 0]
    if index == 2:
        list =  [0, 0, vol[vol_index], 0, 0, 0]
    if index == 3:
        list =  [0, 0, 0, vol[vol_index], 0, 0]
    if index == 4:
        list =  [0, 0, 0, 0, vol[vol_index], 0]
    if index == 5:
        list =  [0, 0, 0, 0, 0, vol[vol_index]]
    return list
def get_All_label_brain_sameAgeRange_size():
    All_label_brain_sameAgeRange_size=[
         [244161, 249064, 248077, 243648, 229577, 240404],
         [229271, 222265, 214511, 204848, 198633, 185513],
         [6278, 7060, 8101, 8110, 11815, 14670],
         [171, 188, 213, 239, 324, 398],
         [16067, 16465, 16988, 15706, 14700, 13895],
         [56659, 55922, 54248, 52592, 50812, 49737],
         [8901, 8725, 8372, 7967, 7540, 7138],
         [3511, 3391, 3263, 3094, 3073, 3164],
         [4808, 4727, 4551, 4330, 4148, 4201],
         [2137, 2022, 2003, 1972, 1850, 1855],
         [819, 889, 921, 1035, 1239, 1451],
         [1599, 1602, 1530, 1585, 1628, 1639],
         [21344, 21678, 21619, 21645, 20648, 20658],
         [4397, 4380, 4340, 4322, 4119, 3978],
         [1542, 1564, 1514, 1510, 1411, 1290],
         [440, 410, 383, 378, 342, 327],
         [4545, 4456, 4349, 4257, 4072, 4066],
         [243409, 249038, 247329, 244231, 229243, 240038],
         [229217, 222120, 214963, 205351, 198451, 185551],
         [5720, 6348, 7248, 7885, 11103, 13714],
         [203, 249, 270, 255, 350, 370],
         [15570, 16076, 16040, 15208, 14272, 13070],
         [57649, 57040, 55675, 53988, 51813, 50204],
         [8067, 7850, 7635, 7373, 7030, 6855],
         [3573, 3446, 3337, 3264, 3245, 3398],
         [4932, 4856, 4654, 4411, 4261, 4223],
         [2166, 2104, 2054, 1961, 1861, 1840],
         [4485, 4459, 4410, 4429, 4265, 4092],
         [1751, 1763, 1703, 1679, 1622, 1507],
         [561, 528, 505, 488, 458, 444],
         [4531, 4459, 4364, 4270, 4117, 4054]]
    return All_label_brain_sameAgeRange_size
labels = [
        2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 26, 28, 41, 42, 43,
        44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60
    ]
def get_cmap(labels):
    colors1 = [
        [0, 0, 0],
        [70, 130, 180],
        [245, 245, 245],
        [205, 62, 78],
        [120, 18, 134],
        [196, 58, 250],
        [0, 148, 0],
        [220, 248, 164],
        [230, 148, 34],
        [0, 118, 14],
        [0, 118, 14],
        [122, 186, 220],
        [236, 13, 176],
        [12, 48, 255],
        [204, 182, 142],
        [42, 204, 164],
        [119, 159, 176],
        [220, 216, 20],
        [103, 255, 255],
        [80, 196, 98],
        [60, 58, 210],
        [60, 58, 210],
        [60, 58, 210],
        [60, 58, 210],
        [60, 60, 60],
        [255, 165, 0],
        [255, 165, 0],
        [0, 255, 127],
        [165, 42, 42],
        [135, 206, 235],
        [160, 32, 240],
        [0, 200, 200],
        [100, 50, 100],
        [135, 50, 74],
        [122, 135, 50],
        [51, 50, 135],
        [74, 155, 60],
        [120, 62, 43],
        [74, 155, 60],
        [122, 135, 50],
        [70, 130, 180],
        [0, 225, 0],
        [205, 62, 78],
        [120, 18, 134],
        [196, 58, 250],
        [0, 148, 0],
        [220, 248, 164],
        [230, 148, 34],
        [0, 118, 14],
        [0, 118, 14],
        [122, 186, 220],
        [236, 13, 176],
        [13, 48, 255],
        [220, 216, 20],
        [103, 255, 255],
        [80, 196, 98],
        [60, 58, 210],
        [255, 165, 0],
        [255, 165, 0],
        [0, 255, 127],
        [165, 42, 42]
    ]
    colors = [
        [245, 245, 245],
        [205, 62, 78],
        [120, 18, 134],
        [196, 58, 250],
        [220, 248, 164],
        [230, 148, 34],
        [0, 118, 14],
        [122, 186, 220],
        [236, 13, 176],
        [12, 48, 255],
        [204, 182, 142],
        [42, 204, 164],
        [119, 159, 176],
        [220, 216, 20],
        [103, 255, 255],
        [255, 165, 0],
        [165, 42, 42],
        [0, 225, 0],
        [205, 62, 78],
        [120, 18, 134],
        [196, 58, 250],
        [220, 248, 164],
        [230, 148, 34],
        [0, 118, 14],
        [122, 186, 220],
        [236, 13, 176],
        [13, 48, 255],
        [220, 216, 20],
        [103, 255, 255],
        [255, 165, 0],
        [165, 42, 42]
    ]
    colors_normalized = np.array(colors) / 255.0
    n_colors = len(labels)
    color_values = np.linspace(0, 1, n_colors)
    color_list = [(color_values[i], colors_normalized[i]) for i in range(n_colors)]
    cmap_custom = LinearSegmentedColormap.from_list('custom_cmap', color_list)
    return cmap_custom
