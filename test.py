from flask import Flask, render_template, redirect, request, session, url_for, render_template_string
import tools
import os
from tigersyn.brainage.utils import get_volumes
import numpy as np

app = Flask(__name__)
labels = [
    2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 41, 42, 43, 44, 46, 47, 49,
    50, 51, 52, 53, 54, 58, 60
]


@app.route("/", methods=['GET', 'POST'])
def home():
    # brain_age = 10
    # brain_size = get_volumes(os.path.join('static', 'image_syn.nii.gz'), labels)
    # brain_size = (np.rint(brain_size)).astype(int)

    # vol_data = tools.age_to_json(brain_age, brain_size)

    # return render_template('segmentation.html',
    #                        img_fname="test",
    #                        image_iframe="test",
    #                        mask_iframe="test",
    #                        options=tools.options,
    #                        brainAge=brain_age,
    #                        vol_data=vol_data)

    return render_template('hippo_seg.html',
                           img_fname='img_fn',
                           image_iframe='img_iframe',
                           mask_iframe='mask_iframe',
                           hippo_vol=12345)


if __name__ == '__main__':
    # app.run(debug=True)
    import nibabel as nib
    img = nib.load(r'static\image_syn.nii.gz').get_fdata()
    import matplotlib.pyplot as plt
    plt.imshow((img == 24)[100, :, :])
    plt.show()
