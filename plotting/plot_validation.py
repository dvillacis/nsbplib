import argparse
import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tikzplotlib
from pathlib import Path
from nsbplib.solvers.rof.solver import ROFSolver_2D
from nsbplib.operators.Gradient import Gradient
from nsbplib.operators.Patch import Patch
from nsbplib.data_generation import load_training_data

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def denoise(noisy_imgs,param,px,py):
    nx,ny = noisy_imgs[0].shape
    recons = np.zeros(shape=(nx,ny,len(noisy_imgs)))
    K = Gradient(dims=((nx,ny)))
    for i,noisy_img in enumerate(noisy_imgs):
        solver = ROFSolver_2D(noisy_img,K)
        data_par = Patch(param,px,py)
        reg_par = Patch(np.ones(px*py),px,py)
        recons[:,:,i] = solver.solve(data_par=data_par,reg_par=reg_par)
    return recons

parser = argparse.ArgumentParser(description='Plot validation results')
parser.add_argument('dataset_dir', type=str, help='Path to validation dataset directory')
parser.add_argument('results_dirs', type=str, nargs='+', help='Path to results directories')
args = parser.parse_args()

dataset_dir = Path(args.dataset_dir)
if dataset_dir.exists() == False:
    print(f'Dataset directory does not exist: {dataset_dir}')
    sys.exit(1)

plots_output_dir = Path('experiments_output/plots')
if plots_output_dir.exists() == False:
    raise ValueError(f'Experiments output directory does not exist: {plots_output_dir}')

num_val_imgs, true_imgs, noisy_imgs = load_training_data(dataset_dir,num_training_data=10)

summary = pd.DataFrame(columns=['Patch Size','Dataset Size','MPSNR','MSSIM'])
for results_dir in args.results_dirs:
    print(results_dir)
    results_dir = Path(results_dir)
    if results_dir.exists() == False:
        print(f'Results directory does not exist: {results_dir}')
        sys.exit(1)
    for ts_folder in os.listdir(results_dir):
        base = os.path.basename(results_dir)
        print(f'{base}_{ts_folder}')
        px = 1
        py = 1
        if len(base.split('_')) == 4:
            px = int(base.split('_')[2][2:])
            py = int(base.split('_')[3][2:])
        param = np.load(results_dir / ts_folder / f'{base}_optimal_par.npy')
        recons = denoise(noisy_imgs=noisy_imgs,param=param,px=px,py=py)
        l2_recs = []
        psnr_recs = []
        ssim_recs = []
        for i in range(recons.shape[2]):
            l2_recs.append(0.5 * np.linalg.norm(true_imgs[i].ravel() - recons[:,:,i].ravel())**2)
            ssim_recs.append(ssim(true_imgs[i],recons[:,:,i],data_range=recons[:,:,i].max() - recons[:,:,i].min()))
            psnr_recs.append(psnr(true_imgs[i],recons[:,:,i]))
        # print(f'px:{px} py:{py} ts:{ts_folder} MPSNR:{np.mean(psnr_recs)} MSSIM:{np.mean(ssim_recs)}')
        summary = pd.concat([summary,pd.DataFrame([[int(px),int(ts_folder),np.mean(psnr_recs),np.mean(ssim_recs)]],columns=['Patch Size','Dataset Size','MPSNR','MSSIM'])],ignore_index=True)


print(summary)
# Plotting Dataframe
fig, ax = plt.subplots()
for patch_group,patch_group_data in summary.groupby('Patch Size'):
    patch_group_data = patch_group_data.sort_values(by=['Dataset Size'])
    ax.plot(patch_group_data['Dataset Size'],patch_group_data['MPSNR'],label=f'Patch Size {patch_group}',marker='o')
    
ax.set_title('Validation MPSNR vs Dataset Size')
ax.set_xlabel('Dataset Size')
ax.set_ylabel('Validation MPSNR')
# ax.legend()     

# plt.show()
tikzplotlib.save(f'{plots_output_dir}/training_mpsnr_vs_dataset_size.tex')
