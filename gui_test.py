import numpy as np
import torch
import torch.nn
import nibabel as nib
from nilearn.image import reorder_img
import onnxruntime as ort
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use('TkAgg')
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.title("test")
root.geometry("1200x700")
root.resizable(0,0)

def turnDataToInputData(file_path):
    origin = nib.load(file_path)
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

def get_path():
    global text
    text = filedialog.askopenfilename()

btn = tk.Button(root,
                text='點擊開啟要切割的檔案',
                font=('Arial',20,'bold'),
                command=get_path
              )
btn.pack()

def change_x(self):
    x = s1.get()
    x = int(x*x_limit/100)
    plt.subplot(2, 3, 1)
    plt.imshow(np.rot90(input_data[0, 0, x, ...]), cmap='gray')
    plt.subplot(2, 3, 4)
    plt.imshow(np.rot90(output[0, 0, x, ...]), cmap='gray')
    canvas.draw()
def change_y(self):
    y = s2.get()
    y = int(y * y_limit / 100)
    plt.subplot(2, 3, 2)
    plt.imshow(np.rot90(input_data[0, 0, :, y, :]), cmap='gray')
    plt.subplot(2, 3, 5)
    plt.imshow(np.rot90(output[0, 0, :, y, :]), cmap='gray')
    canvas.draw()
def change_z(self):
    z = s3.get()
    z = int(z * z_limit / 100)
    plt.subplot(2, 3, 3)
    plt.imshow(np.rot90(input_data[0, 0, :, :, z]), cmap='gray')
    plt.subplot(2, 3, 6)
    plt.imshow(np.rot90(output[0, 0, :, :, z]), cmap='gray')
    canvas.draw()

def show():
    global input_data,output
    model = "hippo.onnx"                                               # model data path
    input_data = turnDataToInputData(text)                             # path that want to segmentation "candi_oasis_aseg/raw123/CANDI_BPDwoPsy_030.nii.gz"
    output = predict(model, input_data, False)

    x,y,z = 60,60,60
    fig = plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(np.rot90(input_data[0, 0, x, ...]), cmap='gray')
    plt.subplot(2, 3, 2)
    plt.imshow(np.rot90(input_data[0, 0, :, y, :]), cmap='gray')
    plt.subplot(2, 3, 3)
    plt.imshow(np.rot90(input_data[0, 0, :, :, z]), cmap='gray')
    plt.subplot(2, 3, 4)
    plt.imshow(np.rot90(output[0, 0, x, ...]), cmap='gray')
    plt.subplot(2, 3, 5)
    plt.imshow(np.rot90(output[0, 0, :, y, :]), cmap='gray')
    plt.subplot(2, 3, 6)
    plt.imshow(np.rot90(output[0, 0, :, :, z]), cmap='gray')
    global canvas
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    global x_limit, y_limit, z_limit
    x_limit = output.shape[2]-1
    y_limit = output.shape[3]-1
    z_limit = output.shape[4]-1
    global s1,s2,s3
    s1 = tk.Scale(root, from_=0, to=100, orient='horizontal', showvalue=0, resolution=1, command=change_x)
    s1.place(anchor='s',x=290,y=700)
    s2 = tk.Scale(root, from_=0, to=100, orient='horizontal', showvalue=0, resolution=1, command=change_y)
    s2.place(anchor='s',x=610,y=700)
    s3 = tk.Scale(root, from_=0, to=100, orient='horizontal', showvalue=0, resolution=1, command=change_z)
    s3.place(anchor='s',x=945,y=700)
btn2 = tk.Button(root,
                text='點擊查看切割結果',
                font=('Arial',20,'bold'),
                command=show
              )
btn2.pack()
tk.mainloop()