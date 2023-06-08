import sys, os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from nsbplib.experiment_utils import read_stats_file

parser = argparse.ArgumentParser(description='Generate table of perfomance metrics')
parser.add_argument('results_dirs', type=str, nargs='+', help='Path to the results directory')

args = parser.parse_args()

table_output_dir = Path('experiments_output/tables')
if table_output_dir.exists() == False:
    raise ValueError(f'Experiments output directory does not exist: {table_output_dir}')

summary = pd.DataFrame(columns=['PS','2','4','6','8','10','12','14','16','18','20','22','24','26','28','30'])

for results_dir in args.results_dirs:
    print(f'Processing results in {results_dir}')
    results_dir = Path(results_dir)
    if results_dir.exists() == False:
        print(f'Results directory does not exist: {results_dir}')
        sys.exit(1)
    base = os.path.basename(results_dir)
    px = 1
    py = 1
    if len(base.split('_')) >= 4:
        px = int(base.split('_')[2][2:])
        py = int(base.split('_')[3][2:])
    print(f'{px=}, {py=}')
    summary_row = [f'{px}x{py}']
    print(os.listdir(results_dir))
    for ts_folder in sorted(os.listdir(results_dir),key=int):
        print(ts_folder)
        n_reg_jev,n_fev,nit,njev = read_stats_file(results_dir / f'{ts_folder}'/ f'{base}_stats.txt')
        summary_row.append(f'{nit}/{njev}/{n_reg_jev}')
    print(summary_row)
    summary.loc[len(summary)] = summary_row

print(summary)
with open(table_output_dir / 'performance_table.txt','w') as f:
    f.write(summary.to_latex(index=False))