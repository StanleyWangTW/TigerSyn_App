import nibabel as nib
import pandas as pd
from jinja2 import Template
from tigersyn.brainage.utils import get_volumes
import tools

labels = [
    2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 26, 28, 41, 42, 43, 44, 46, 47, 49, 50,
    51, 52, 53, 54, 58, 60
]

label_to_name = {
    2: 'Left Cerebral WM',
    41: 'Right Cerebral WM',
    3: 'Left Cerebral Cortex',
    42: 'Right Cerebral Cortex',
    4: 'Left Lateral Ventricle',
    43: 'Right Lateral Ventricle',
    5: 'Left Inf Lat Vent',
    44: 'Right Inf Lat Vent',
    7: 'Left Cerebellum WM',
    46: 'Right Cerebellum WM',
    8: 'Left Cerebellum Cortex',
    47: 'Right Cerebellum Cortex',
    10: 'Left Thalamus',
    49: 'Right Thalamus',
    11: 'Left Caudate',
    50: 'Right Caudate',
    12: 'Left Putamen',
    51: 'Right Putamen',
    13: 'Left Pallidum',
    52: 'Right Pallidum',
    14: '3rd Ventricle',
    53: 'Right Hippocampus',
    15: '4th Ventricle',
    54: 'Right Amygdala',
    16: 'Brain Stem',
    58: 'Right Accumbens area',
    17: 'Left Hippocampus',
    60: 'Right VentralDC',
    18: 'Left Amygdala',
    62: 'Right vessel',
    24: 'CSF',
    63: 'Right choroid plexus',
    26: 'Left Accumbens area',
    77: 'WM hypointensities',
    28: 'Left VentralDC',
    85: 'Optic Chiasm',
    30: 'Left vessel',
    251: 'CC Posterior',
    31: 'Left choroid plexus',
    252: 'CC Mid Posterior',
    253: 'CC Central',
    254: 'CC Mid Anterior',
    255: 'CC Anterior'
}

from fpdf import FPDF


class PDF(FPDF):

    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'My Table Report', 0, 1, 'C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, 'Page %s' % self.page_no(), 0, 0, 'C')

    def add_table(self, header, data):
        self.set_font('Arial', '', 12)

        # Add a header row
        for col in header:
            self.cell(50, 10, col, 1, align='C')
        self.ln()

        # Add data rows
        for row in data:
            for col in row:
                self.cell(50, 10, str(col), 1, align='C')
            self.ln()


def create_data(img_f, mask_f):
    vols = get_volumes(mask_f, labels)

    img = nib.load(img_f).get_fdata()
    tools.img_save3GrayScale(img, r'.')
    mask = nib.load(mask_f).get_fdata()
    tools.mask_to_3img(mask, '.', 'gist_ncar')

    data = list()
    for idx, v in enumerate(vols):
        data.append([label_to_name[labels[idx]], round(v)])

    return data


# Create instance of FPDF class
pdf = PDF()
pdf.add_page()

# Table header and data
table_header = ['Label Name', 'Volume']
img_f = r'CANDI_BPDwoPsy_030_1mm.nii.gz'
mask_f = r'CC0001_philips_15_55_M_aseg.nii.gz'
table_data = create_data(img_f, mask_f)

# Add table to PDF
pdf.add_table(table_header, table_data)

# Save the pdf with name .pdf
pdf.output("table_report.pdf")
