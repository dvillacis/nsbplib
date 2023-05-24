# Nonsmooth Bilevel TV Learning

A cleaned up version of the code for papers: 

* De los Reyes, Juan Carlos, and David Villacís. "Optimality conditions for bilevel imaging learning problems with total variation regularization." SIAM Journal on Imaging Sciences 15.4 (2022): 1646-1689.
* De los Reyes, Juan Carlos, and David Villacís. "Interpretable Model Learning in Variational Imaging: A Bilevel Optimization Approach"

## Prerequisites
The python environment must have numpy, pylops and pyproximal installed.

## Installing a local version
After cloning the project, cd in the root folder and install the module using pip in experimental mode

```bash
$ cd nsbplib
$ pip install -e .
```

## Running prebuilt experiments

* Learning Optimal Scalar Parameter (Data Learning)
```bash
$ cd experiments
$ python learn_optimal_scalar_data_parameter.py $dataset_name $output_folder --size_training_set $size_dataset
```

* Learning Optimal Patch Parameter (Data Learning)
```bash
$ cd experiments
$ python learn_optimal_patch_data_parameter.py $dataset_name $output_folder --patch_size $patch_size --size_training_set $size_dataset
```

* Learning Optimal Scalar Parameter (Regularization Learning)
```bash
$ cd experiments
$ python learn_optimal_scalar_reg_parameter.py $dataset_name $output_folder --size_training_set $size_dataset
```

* Learning Optimal Patch Parameter (Regularization Learning)
```bash
$ cd experiments
$ python learn_optimal_patch_reg_parameter.py $dataset_name $output_folder --patch_size $patch_size --size_training_set $size_dataset
```

## Plotting the results from output_folder
For regenerating the plots presented in the paper, there are several scripts that generate the plots and tables

* Plotting scalar cameraman reconstruction
```bash
$ python plotting/plot_scalar_reconstruction $output_folder/$dataset_name
```

* Plotting reconstruction from different models
```bash
$ python plotting/plot_reconstructions $output_folder_1 $output_folder_2 ...
```

## Plotting the validation results
This script generates the plot regarding the validation error of the learned parameter for different patch sizes and different training set sizes.

```bash
$ python plotting/plot_validation $validation_dataset_path $output_folder_1 $output_folder_2 ...