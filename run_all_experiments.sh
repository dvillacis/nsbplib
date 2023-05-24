#!/bin/bash

# Run all experiments
# Usage: ./run_all_experiments.sh

for sz in 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30
do
    echo "Running experiment with training set size $sz and patch size 1"
    if [ ! -d "experiments_output/faces_small/$sz" ]; then 
        python experiments/learn_optimal_scalar_data_parameter.py datasets/faces_small experiments_output --size_training_set $sz
    else
        echo "Experiment already executed, skipping..."
    fi

    for ps in 2 4 8 16 32
    do
        echo "Running experiment with training set size $sz and patch size $ps"
        if [ ! -d "experiments_output/faces_small_px${ps}_py${ps}/$sz" ]; then
            python experiments/learn_optimal_patch_data_parameter.py datasets/faces_small experiments_output --size_training_set $sz --patch_size $ps
        else
            echo "Experiment already executed, skipping..."
        fi
    done
done