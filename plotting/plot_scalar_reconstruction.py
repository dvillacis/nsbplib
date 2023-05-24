import sys
import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Parse arguments
parser = argparse.ArgumentParser(description='Plot scalar reconstruction')
parser.add_argument('results_dir', type=str, help='Path to the results directory')

args = parser.parse_args()

results_dir = Path(args.results_dir)
if results_dir.exists() == False:
    print(f'Results directory does not exist: {results_dir}')
    sys.exit(1)

# Load the results
basename = results_dir.parent.name
evals = pd.read_pickle(results_dir / f'{basename}_evals.pkl')
noisy_imgs = np.load(results_dir / f'{basename}_noisy_imgs.npy')
true_imgs = np.load(results_dir / f'{basename}_true_imgs.npy')
recons = np.load(results_dir / f'{basename}_recons.npy')
optimal_par = np.load(results_dir / f'{basename}_optimal_par.npy')

# Plot the results
nx,ny,nz = true_imgs.shape
fig,ax = plt.subplots(nz,3,figsize=(12,4))

if nz == 1:
    ax[0].imshow(true_imgs[:,:,0],cmap='gray')
    ax[0].set_title('True image')
    ax[0].set_xticklabels([])
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_yticklabels([])
    ax[1].imshow(noisy_imgs[:,:,0],cmap='gray')
    ax[1].set_title('Noisy image')
    ax[1].set_xlabel(f'PSNR={psnr(true_imgs[:,:,0],noisy_imgs[:,:,0]):.4f} SSIM={ssim(true_imgs[:,:,0],noisy_imgs[:,:,0],data_range=noisy_imgs[:,:,0].max() - noisy_imgs[:,:,0].min()):.4f}')
    ax[1].set_xticklabels([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_yticklabels([])
    ax[2].imshow(recons[:,:,0],cmap='gray')
    ax[2].set_title(f'Reconstruction')
    ax[2].set_xlabel(f'PSNR={psnr(true_imgs[:,:,0],recons[:,:,0]):.4f} SSIM={ssim(true_imgs[:,:,0],recons[:,:,0],data_range=recons[:,:,0].max() - recons[:,:,0].min()):.4f}')
    ax[2].set_xticklabels([])
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    ax[2].set_yticklabels([])
else:
    for i in range(nz):
        print(true_imgs.shape,noisy_imgs.shape,recons.shape)
        ax[i][0].imshow(true_imgs[:,:,i],cmap='gray')
        if i == 0: ax[i,0].set_title('True image')
        ax[i,0].set_xticklabels([])
        ax[i,0].set_xticks([])
        ax[i,0].set_yticks([])
        ax[i,0].set_yticklabels([])
        ax[i,1].imshow(noisy_imgs[:,:,i],cmap='gray')
        if i == 0: ax[i,1].set_title('Noisy image')
        ax[i,1].set_xticklabels([])
        ax[i,1].set_xticks([])
        ax[i,1].set_yticks([])
        ax[i,1].set_yticklabels([])
        ax[i,1].set_xlabel(f'PSNR={psnr(true_imgs[:,:,i],noisy_imgs[:,:,i]):.4f} SSIM={ssim(true_imgs[:,:,i],noisy_imgs[:,:,i],data_range=noisy_imgs[:,:,i].max() - noisy_imgs[:,:,i].min()):.4f}')
        ax[i,2].imshow(recons[:,:,i],cmap='gray')
        if i == 0: ax[i,2].set_title(f'Reconstruction')
        ax[i,2].set_xticklabels([])
        ax[i,2].set_xticks([])
        ax[i,2].set_yticks([])
        ax[i,2].set_yticklabels([])
        ax[i,2].set_xlabel(f'PSNR={psnr(true_imgs[:,:,i],recons[:,:,i]):.4f} SSIM={ssim(true_imgs[:,:,i],recons[:,:,i],data_range=recons[:,:,i].max() - recons[:,:,i].min()):.4f}')
plt.show()

