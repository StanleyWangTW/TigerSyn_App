import os

import numpy as np
import cv2
import matplotlib.pyplot as plt

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