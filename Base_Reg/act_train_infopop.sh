#!/bin/bash
#SBATCH --job-name=train_infopop
#SBATCH -A research
#SBATCH -c 10
#SBATCH -o act_out_train_infopop.out
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END

python debug_infopop.py -device 0 -batch_size 32 -model CNN_RNN_Regression_act -seed 1 -save_dir checkpoints/CNNR_act_infopop.pt