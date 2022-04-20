#!/bin/bash
#SBATCH --job-name=train_infopop
#SBATCH -A research
#SBATCH -c 10
#SBATCH -o out_train_infopop.out
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END

python train_infopop.py -device 0 -batch_size 256 -model CNN_RNN_Regression -seed 1 -save_dir checkpoints/CNN_infopop.pt -report_every 100 -epochs 15
#python train_infopop.py -device 0 -batch_size 64 -model CNN_RNN_Regression_act -seed 1 -save_dir checkpoints/CNNR_infopop.pt -report_every 100 -epochs 15
