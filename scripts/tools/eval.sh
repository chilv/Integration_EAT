#!/bin/bash
for rate in $(seq 0 9)
do
    echo $rate
    for joint in $(seq 0 11)
    do
        # joint=10
        echo "Evaluation on: Rate_idx: $rate, Joint: $joint"
        python scripts/model_evaluation.py --experiment_name "test_save_best_flawed_1" --joint $joint --rate_idx $rate --sim_device cuda:1
    done
done
