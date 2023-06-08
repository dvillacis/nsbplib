import sys
import argparse
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import tikzplotlib

parser = argparse.ArgumentParser(description='Plot learned parameters')
parser.add_argument('results_dirs', type=str, nargs='+', help='Path to the results directory')

args = parser.parse_args()

plots_output_dir = Path('experiments_output/plots')
if plots_output_dir.exists() == False:
    raise ValueError(f'Experiments output directory does not exist: {plots_output_dir}')
print(args.results_dirs)
fig,ax = plt.subplots(1,len(args.results_dirs),figsize=(12,4))

for j,results_dir in enumerate(args.results_dirs):
    print(f'Plotting parameter in {results_dir}')
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
    true_imgs = np.load(results_dir / f'{basename}_true_imgs.npy')
    nx,ny,nz = true_imgs.shape
    param = np.load(results_dir / f'{basename}_optimal_par.npy')
    m = nx // px
    param = np.kron(param.reshape((px,py)),np.ones((m,m)))
    p = ax[j].imshow(param)
    cb = plt.colorbar(p,orientation='horizontal',ax=ax[j])
    ax[j].set_title(f'{px}x{py}')
    ax[j].axis('off')
plt.show()
# tikzplotlib.save(f'{plots_output_dir}/optimal_patches.tex')