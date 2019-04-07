"""
Quick Python script for saving .png files from the crop_imag.mat images.
"""

import numpy as np
from scipy import io
from scipy.misc import imsave
import os
from PIL import Image
#from matplotlib import pyplot as plt

directory = # 

def save_image(folder_name, extension=".jpg"):
    try:
        img = np.array((io.loadmat('crop_image.mat', appendmat=False))['dxImage']['img'][0][0])
        img_compressed = (img*255.0/np.max(img)).astype('uint8')
        rgb_img = np.asarray(Image.fromarray(img_compressed).convert('RGB'))
        imsave(folder_name + extension, rgb_img)
    except FileNotFoundError:
        try:
            #print("Running...")
            img = np.array((io.loadmat('crop_image_fracturemask.mat', appendmat=False))['dxImage']['img'][0][0])
            img_compressed = (img * 255.0 / np.max(img)).astype('uint8')
            rgb_img = np.asarray(Image.fromarray(img_compressed).convert('RGB'))
            imsave(folder_name + extension, rgb_img)
        except FileNotFoundError:
            print("Could not locate mat file in folder " + folder_name)

for subdir, dirs, files in os.walk(directory):
    #print(dirs)
    for folder in dirs:
        os.chdir(directory + "\\" + folder)
        #print(folder)
        save_image(folder, ".png")
