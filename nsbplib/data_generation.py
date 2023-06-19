
from argparse import ArgumentError
import numpy as np
from pylops.signalprocessing.convolve2d import Convolve2D
from os import listdir, pardir
from os.path import isfile, join, isdir

from PIL import Image


def load_training_data(ds_dir,num_training_data:int=1):
    true_imgs_dir = join(ds_dir,'true')
    noisy_imgs_dir = join(ds_dir,'noisy')
    if not isdir(true_imgs_dir):
        raise ArgumentError(f'{true_imgs_dir} does not exists...')
    print(f'Loading training data from {true_imgs_dir} and {noisy_imgs_dir}')
    true_imgs = []
    true_imgs_path = [f for f in sorted(listdir(true_imgs_dir)) if isfile(join(true_imgs_dir, f))][:num_training_data]
    for img_path in sorted(true_imgs_path):
        if '.png' in img_path:
            img = np.array(Image.open(join(true_imgs_dir,img_path)).convert('L'))
            img = img / np.amax(img)
            true_imgs.append(img)
        
    noisy_imgs = []
    noisy_imgs_path = [f for f in sorted(listdir(noisy_imgs_dir)) if isfile(join(noisy_imgs_dir, f))][:num_training_data]
    for img_path in sorted(noisy_imgs_path):
        if '.png' in img_path:
            img = np.array(Image.open(join(noisy_imgs_dir,img_path)).convert('L'))
            img = img / np.amax(img)
            noisy_imgs.append(img)
    return len(true_imgs), true_imgs, noisy_imgs

def load_training_data_deblurring(ds_dir,num_training_data:int=1, lost_domain_size=10):
    true_imgs_dir = join(ds_dir,'true')
    noisy_imgs_dir = join(ds_dir,'noisy')
    if not isdir(true_imgs_dir):
        raise ArgumentError(f'{true_imgs_dir} does not exists...')
    print(f'Loading training data from {true_imgs_dir} and {noisy_imgs_dir}')
    true_imgs = []
    true_imgs_path = [f for f in sorted(listdir(true_imgs_dir)) if isfile(join(true_imgs_dir, f))][:num_training_data]
    
    for img_path in sorted(true_imgs_path):
        if '.png' in img_path:
            img = np.array(Image.open(join(true_imgs_dir,img_path)).convert('L'))
            img = img / np.amax(img)
            true_imgs.append(img)
    
    # Convolution Operator 2D
    # Blurring guassian kernel
    nx,ny = true_imgs[0].shape
    nh = [5, 5]
    hz = np.exp(-0.05 * np.linspace(-(nh[0] // 2), nh[0] // 2, nh[0]) ** 2)
    hx = np.exp(-0.05 * np.linspace(-(nh[1] // 2), nh[1] // 2, nh[1]) ** 2)
    hz /= np.trapz(hz)  # normalize the integral to 1
    hx /= np.trapz(hx)  # normalize the integral to 1
    kernel = hz[:, np.newaxis] * hx[np.newaxis, :]
    # kernel = np.array([[0.0, 0.1, 0.0],[0.0, 0.8, 0.0],[0.0, 0.1, 0.0]])
    # print(kernel)
    # Convolution Operator
    ConvOp = Convolve2D((nx, ny),h=kernel, offset=(nh[0] // 2, nh[1] // 2), dtype='float64')
    print(f'Convolution Operator: {ConvOp.shape}')
    noisy_imgs = []
    noisy_imgs_path = [f for f in sorted(listdir(true_imgs_dir)) if isfile(join(true_imgs_dir, f))][:num_training_data]
    for img_path in sorted(noisy_imgs_path):
        if '.png' in img_path:
            img = np.array(Image.open(join(true_imgs_dir,img_path)).convert('L'))
            img = img / np.amax(img)
            blur = (ConvOp*img.ravel()).reshape(nx,ny)
            blur = blur / np.amax(blur)
            blur = np.clip(blur,0,1)
            np.random.seed(1234)
            noise = np.random.normal(loc=0,scale=0.01,size=img.shape)
            noisy = blur+noise
            noisy = noisy/np.amax(noisy)
            noisy = np.clip(noisy,0,1)
            noisy_imgs.append(noisy)
    return len(true_imgs), true_imgs, noisy_imgs, ConvOp
    