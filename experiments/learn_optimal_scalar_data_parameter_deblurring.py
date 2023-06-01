import sys
import os
import argparse
from pathlib import Path

from nsbplib.solvers.nstrbox.solver import solve as nstrbox_solve
from nsbplib.upper_level_problems import UpperScalarDataLearningInpainting
from nsbplib.experiment_utils import load_start_parameter, save_nsbpl_results

# Parse arguments
parser = argparse.ArgumentParser(description='Learn optimal scalar data parameter')
parser.add_argument('dataset_dir', type=str, help='Path to dataset directory')
parser.add_argument('output_dir', type=str, help='Path to output directory')
parser.add_argument('--size_training_set', type=int, default=1, help='Size of training set')
args = parser.parse_args()

dataset_dir = Path(args.dataset_dir)
output_dir = Path(args.output_dir)
if dataset_dir.exists() == False:
    print(f'Dataset directory does not exist: {dataset_dir}')
    sys.exit(1)
if output_dir.exists() == False:
    print(f'Output directory does not exist: {output_dir}')
    sys.exit(1)
if args.size_training_set < 1:
    print(f'Invalid size of training set: {args.size_training_set}')
    sys.exit(1)

size_training_set = args.size_training_set

upper_level_problem = UpperScalarDataLearningInpainting(ds_dir=dataset_dir,num_training_data=size_training_set,verbose=True)

x0 = load_start_parameter(10.0)

# Solve the problem
evals,sol = nstrbox_solve(upper_level_problem,x0,verbose=True)
true_imgs, noisy_imgs, recons = upper_level_problem.get_training_data()
extra_data = {'true_imgs':true_imgs,'noisy_imgs':noisy_imgs,'recons':recons}

# Save the results
save_nsbpl_results(evals=evals,sol=sol,extra_data=extra_data,outfolder=output_dir,run_name=os.path.basename(dataset_dir)+'_inpainting')