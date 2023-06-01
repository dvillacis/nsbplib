import sys
import argparse
from pathlib import Path
import numpy as np
import tikzplotlib
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

parser = argparse.ArgumentParser(description='Plot reconstructions')
parser.add_argument('results_dirs', type=str, nargs='+', help='Path to the results directory')
parser.add_argument('--num_imgs', type=int, default=5, help='Number of images to plot')

args = parser.parse_args()

plots_output_dir = Path('experiments_output/plots')
if plots_output_dir.exists() == False:
    raise ValueError(f'Experiments output directory does not exist: {plots_output_dir}')

print(args.results_dirs)
fig,ax = plt.subplots(args.num_imgs,len(args.results_dirs)+1,figsize=(12,4))

for i in range(args.num_imgs):
    print(f'Plotting image {i}')
    for j,results_dir in enumerate(args.results_dirs):
        print(f'Plotting results in {results_dir}')
        results_dir = Path(results_dir)
        if results_dir.exists() == False:
            print(f'Results directory does not exist: {results_dir}')
            sys.exit(1)
        px = 1
        py = 1
        basename = results_dir.parent.name 
        if len(basename.split('_')) == 4:
            px = int(basename.split('_')[2][2:])
            py = int(basename.split('_')[3][2:])
        # Load the results
        recons = np.load(results_dir / f'{basename}_recons.npy')
        true_imgs = np.load(results_dir / f'{basename}_true_imgs.npy')
        noisy_imgs = np.load(results_dir / f'{basename}_noisy_imgs.npy')
        print(f'{noisy_imgs.shape=}, {i=}')
        if j==0:
            ax[i,j].imshow(noisy_imgs[:,:,i],cmap='gray')
            if i == 0: ax[i,j].set_title('Noisy Image')
            ax[i,j].set_xticklabels([])
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])
            ax[i,j].set_yticklabels([])
            ax[i,j].set_xlabel(f'PSNR={psnr(true_imgs[:,:,i],noisy_imgs[:,:,i]):.4f}\nSSIM={ssim(true_imgs[:,:,i],noisy_imgs[:,:,i],data_range=noisy_imgs[:,:,i].max() - noisy_imgs[:,:,i].min()):.4f}')
        ax[i,j+1].imshow(recons[:,:,i],cmap='gray')
        if i==0: ax[i,j+1].set_title(f'{px}x{py}')
        ax[i,j+1].set_xticklabels([])
        ax[i,j+1].set_xticks([])
        ax[i,j+1].set_yticks([])
        ax[i,j+1].set_yticklabels([])
        ax[i,j+1].set_xlabel(f'PSNR={psnr(true_imgs[:,:,i],recons[:,:,i]):.4f}\nSSIM={ssim(true_imgs[:,:,i],recons[:,:,i],data_range=recons[:,:,i].max() - recons[:,:,i].min()):.4f}')
plt.show()
# tikzplotlib.save(f'{plots_output_dir}/faces_small_recons.tex')