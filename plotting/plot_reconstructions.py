import sys
import argparse
from pathlib import Path
import numpy as np
import tikzplotlib
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage import exposure

def hist_eq(image):
    hist, bins = exposure.histogram(image, nbins=256, normalize=False)
    # append any remaining 0 values to the histogram
    image = (255*image).astype(np.uint8)
    hist = np.hstack((hist, np.zeros((255 - bins[-1])))) 
    cdf = 255*(hist/hist.sum()).cumsum()
    equalized = cdf[image].astype(np.uint8)

    return equalized/np.amax(equalized)

parser = argparse.ArgumentParser(description='Plot reconstructions')
parser.add_argument('results_dirs', type=str, nargs='+', help='Path to the results directory')
parser.add_argument('--num_imgs', type=int, default=1, help='Number of images to plot')

args = parser.parse_args()

plots_output_dir = Path('experiments_output/plots')
if plots_output_dir.exists() == False:
    raise ValueError(f'Experiments output directory does not exist: {plots_output_dir}')

print(args.results_dirs)
fig,ax = plt.subplots(args.num_imgs,len(args.results_dirs)+2,figsize=(12,4))

if args.num_imgs == 1:
    for j,results_dir in enumerate(args.results_dirs):
        print(f'Plotting results in {results_dir}')
        results_dir = Path(results_dir)
        if results_dir.exists() == False:
            print(f'Results directory does not exist: {results_dir}')
            sys.exit(1)
        px = 1
        py = 1
        basename = results_dir.parent.name 
        if len(basename.split('_')) >= 4:
            px = int(basename.split('_')[2][2:])
            py = int(basename.split('_')[3][2:])
        print(f'{px=}, {py=}')
        # Load the results
        recons = np.load(results_dir / f'{basename}_recons.npy')
        true_imgs = np.load(results_dir / f'{basename}_true_imgs.npy')
        noisy_imgs = np.load(results_dir / f'{basename}_noisy_imgs.npy')
        print(f'{noisy_imgs.shape=}')
        # true_img = (true_imgs[:,:,0]-np.amin(true_imgs[:,:,0])/(np.amax(true_imgs[:,:,0])-np.amin(true_imgs[:,:,0])))
        true_img = true_imgs[:,:,0]
        noisy_img = (noisy_imgs[:,:,0]-np.amin(noisy_imgs[:,:,0])/(np.amax(noisy_imgs[:,:,0])-np.amin(noisy_imgs[:,:,0])))
        if j==0:
            ax[0].imshow(true_imgs[:,:,0],cmap='gray')
            ax[0].set_title('True Image')
            ax[0].set_xticklabels([])
            ax[0].set_xticks([])
            ax[0].set_yticks([])
            ax[0].set_yticklabels([])
            # ax[0].set_xlabel(f'PSNR={psnr(true_imgs[:,:,0],noisy_imgs[:,:,0]):.4f}\nSSIM={ssim(true_imgs[:,:,0],noisy_imgs[:,:,0],data_range=noisy_imgs[:,:,0].max() - noisy_imgs[:,:,0].min()):.4f}')

            ax[1].imshow(noisy_imgs[:,:,0],cmap='gray')
            ax[1].set_title('Noisy Image')
            ax[1].set_xticklabels([])
            ax[1].set_xticks([])
            ax[1].set_yticks([])
            ax[1].set_yticklabels([])
            ax[1].set_xlabel(f'PSNR={psnr(true_img,noisy_img):.4f}\nSSIM={ssim(true_img,noisy_img,data_range=noisy_img.max() - noisy_img.min()):.4f}')
        # recons = (recons[:,:,0]-np.amin(recons[:,:,0])/(np.amax(recons[:,:,0])-np.amin(recons[:,:,0])))
        recons = np.clip(recons,0,1)
        recons = recons/np.amax(recons)
        recons = exposure.match_histograms(recons[:,:,0],true_img)
        recons = recons/np.amax(recons)
        ax[j+2].imshow(recons[56:96,56:96],cmap='gray')
        ax[j+2].set_title(f'{px}x{py}')
        ax[j+2].set_xticklabels([])
        ax[j+2].set_xticks([])
        ax[j+2].set_yticks([])
        ax[j+2].set_yticklabels([])
        ax[j+2].set_xlabel(f'PSNR={psnr(true_img,recons):.4f}\nSSIM={ssim(true_img,recons,data_range=true_img.max() - true_img.min()):.4f}')
else:
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
# plt.show()
tikzplotlib.save(f'{plots_output_dir}/faces_medium_deblurring_recons.tex')