import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from nilearn import plotting, image
from flask import render_template_string
import json
from matplotlib.colors import ListedColormap
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps

labels = [
    0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 41, 42, 43, 44, 46, 47, 49,
    50, 51, 52, 53, 54, 58, 60
]


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


def view1image(path1, colormap):
    img1 = image.load_img(path1)

    if colormap:
        view = plotting.view_img(img1,
                                 black_bg=True,
                                 colorbar=False,
                                 bg_img=False,
                                 symmetric_cmap=False,
                                 resampling_interpolation='nearest',
                                 cmap=get_cmap())
    else:
        view = plotting.view_img(img1,
                                 black_bg=True,
                                 colorbar=False,
                                 bg_img=False,
                                 cmap=nilearn_cmaps["brown_blue"])  # ,cmap=plt.cm.get_cmap('gray')

    return view.get_iframe()


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
    All_label_brain_sameAgeRange_size = [[244161, 249064, 248077, 243648, 229577, 240404],
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
                                         [1542, 1564, 1514, 1510, 1411, 1290], [0, 0, 0, 0, 0, 0],
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


def age_to_json(age, brain_size):
    sameAgeRange_size = get_All_label_brain_sameAgeRange_size()
    age_range = get_brain_age_range(age)

    data = list()
    for i in range(len(brain_size)):
        row = dict()
        row['region'] = labels[i + 1]
        row['vol'] = brain_size[i].item()
        row['avg_vol'] = sameAgeRange_size[i]
        row['age_range'] = age_range

        data.append(row)

    return json.dumps(data, ensure_ascii=False)


def get_cmap():
    # custom cmap for labels plotting
    rgb_colors = [(0, 0, 0), (245, 245, 245), (205, 62, 78), (120, 18, 134), (196, 58, 250),
                  (220, 248, 164), (230, 148, 34), (0, 118, 14), (122, 186, 220), (236, 13, 176),
                  (12, 48, 255), (204, 182, 142), (42, 204, 164), (119, 159, 176), (220, 216, 20),
                  (103, 255, 255), (60, 60, 60), (255, 165, 0), (165, 42, 42), (245, 245, 245),
                  (205, 62, 78), (120, 18, 134), (196, 58, 250), (220, 248, 164), (230, 148, 34),
                  (0, 118, 14), (122, 186, 220), (236, 13, 176), (13, 48, 255), (220, 216, 20),
                  (103, 255, 255), (255, 165, 0), (165, 42, 42)]

    # Convert RGB colors to the range [0, 1]
    colors = np.array(rgb_colors) / 255.0
    colors = np.zeros([labels[-1] + 1, 3])
    for idx, l in enumerate(labels):
        colors[l] = rgb_colors[idx]
        colors[l] /= 255.0

    # Create a ListedColormap
    return ListedColormap(colors)


options = [{
    "num": 2,
    "region": "Left Cerebral WM"
}, {
    "num": 3,
    "region": "Left Cerebral Cortex"
}, {
    "num": 4,
    "region": "Left Lateral Ventricle"
}, {
    "num": 5,
    "region": "Left Inf Lat Vent"
}, {
    "num": 7,
    "region": "Left Cerebellum WM"
}, {
    "num": 8,
    "region": "Left Cerebellum Cortex"
}, {
    "num": 10,
    "region": "Left Thalamus"
}, {
    "num": 11,
    "region": "Left Caudate"
}, {
    "num": 12,
    "region": "Left Putamen"
}, {
    "num": 13,
    "region": "Left Pallidum"
}, {
    "num": 14,
    "region": "3rd Ventricle"
}, {
    "num": 15,
    "region": "4th Ventricle"
}, {
    "num": 16,
    "region": "Brain Stem"
}, {
    "num": 17,
    "region": "Left Hippocampus"
}, {
    "num": 18,
    "region": "Left Amygdala"
}, {
    "num": 24,
    "region": "CSF"
}, {
    "num": 26,
    "region": "Left Accumbens area"
}, {
    "num": 28,
    "region": "Left Accumbens area"
}, {
    "num": 41,
    "region": "Right Cerebral WM"
}, {
    "num": 42,
    "region": "Right Cerebral Cortex"
}, {
    "num": 43,
    "region": "Right Lateral Ventricle"
}, {
    "num": 44,
    "region": "Right Inf Lat Vent"
}, {
    "num": 46,
    "region": "Right Cerebellum WM"
}, {
    "num": 47,
    "region": "Right Cerebellum Cortex"
}, {
    "num": 49,
    "region": "Right Thalamus"
}, {
    "num": 50,
    "region": "Right Caudate"
}, {
    "num": 51,
    "region": "Right Putamen"
}, {
    "num": 52,
    "region": "Right Pallidum"
}, {
    "num": 53,
    "region": "Right Hippocampus"
}, {
    "num": 54,
    "region": "Right Amygdala"
}, {
    "num": 58,
    "region": "Right Accumbens area"
}, {
    "num": 60,
    "region": "Right VentralDC"
}]
