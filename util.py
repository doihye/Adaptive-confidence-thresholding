from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect
import re
import numpy as np
import os
import collections
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import matplotlib.cm as cm

def tensor2im(image_tensor, imtype=np.uint8):

    image_numpy = image_tensor.data.cpu().numpy()
    if len(list(image_tensor.size())) == 4:
        image_numpy = np.squeeze(image_numpy, 0)
    image_numpy = (np.transpose(image_numpy, (1,2,0)) + 1) / 2.0 * 255.0
    image_numpy = np.squeeze(image_numpy)

    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path): 

    image_numpy = np.squeeze(image_numpy).astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def save_image_(image_numpy, image_path): #save as plasma mode
    image_numpy = np.squeeze(image_numpy).astype(np.uint8) #
    #image_numpy = (721.5377 * 0.54)/image_numpy
    h,w = image_numpy.shape

    fig = plt.figure()
    fig.set_size_inches(w/h, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image_numpy, cmap='plasma')
    plt.savefig(image_path, dpi = 300)
    plt.close()

import  cv2
def save_image2_(image_numpy, image_path): #save as original mode
    image_numpy = np.squeeze(image_numpy).astype(np.uint8) #
    h,w = image_numpy.shape[0],image_numpy.shape[1]

    fig = plt.figure()
    fig.set_size_inches(w/h, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image_numpy)
    plt.savefig(image_path, dpi = 300)
    plt.close()

def save_image__(image_numpy, image_path): #save as gray mode

    image_numpy = np.squeeze(image_numpy).astype(np.uint8) #
    h,w = image_numpy.shape

    fig = plt.figure()
    fig.set_size_inches(w/h, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image_numpy, cmap='Greys')
    plt.savefig(image_path, dpi = 300)
    plt.close()