
from argparse import ArgumentError
import numpy as np
from pylops import Restriction
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
    
    # Restriction operator
    nx,ny = true_imgs[0].shape
    start_row = nx//3 - lost_domain_size
    end_row = nx//3 + lost_domain_size
    start_col = ny//3 - lost_domain_size
    end_col = ny//3 + lost_domain_size
    mask = np.arange(nx*ny).reshape(nx,ny)
    mask[start_row:end_row,start_col:end_col] = -1
    R = Restriction(nx*ny,mask[np.where(mask >= 0)],dtype=np.float64)
     
    noisy_imgs = []
    noisy_imgs_path = [f for f in sorted(listdir(noisy_imgs_dir)) if isfile(join(noisy_imgs_dir, f))][:num_training_data]
    for img_path in sorted(noisy_imgs_path):
        if '.png' in img_path:
            img = np.array(Image.open(join(noisy_imgs_dir,img_path)).convert('L'))
            img = img / np.amax(img)
            img = (R.T*(R*img.ravel())).reshape(nx,ny)
            noisy_imgs.append(img)
    return len(true_imgs), true_imgs, noisy_imgs, R
    