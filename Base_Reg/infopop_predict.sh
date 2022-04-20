#!/bin/bash
#SBATCH --job-name=pred_summ
#SBATCH -A research
#SBATCH -c 10
#SBATCH -o pred_summ.out
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --Mail-type=END

pt=CNN_infopop.ptCNN_RNN_Regression_seed_1.pt

python main_infopop.py -batch_size 1 -device 0 -predict -startind 0 -endind 1000 -load_dir checkpoints/$pt -filename ../CNN_DailyMail/test.json
python main_infopop.py -batch_size 1 -device 0 -predict -startind 1000 -endind 2000 -load_dir checkpoints/$pt -filename ../CNN_DailyMail/test.json
python main_infopop.py -batch_size 1 -device 0 -predict -startind 2000 -endind 3000 -load_dir checkpoints/$pt -filename ../CNN_DailyMail/test.json
python main_infopop.py -batch_size 1 -device 0 -predict -startind 3000 -endind 4000 -load_dir checkpoints/$pt -filename ../CNN_DailyMail/test.json
python main_infopop.py -batch_size 1 -device 0 -predict -startind 4000 -endind 5000 -load_dir checkpoints/$pt -filename ../CNN_DailyMail/test.json
python main_infopop.py -batch_size 1 -device 0 -predict -startind 5000 -endind 6000 -load_dir checkpoints/$pt -filename ../CNN_DailyMail/test.json
python main_infopop.py -batch_size 1 -device 0 -predict -startind 6000 -endind 7000 -load_dir checkpoints/$pt -filename ../CNN_DailyMail/test.json
python main_infopop.py -batch_size 1 -device 0 -predict -startind 7000 -endind 8000 -load_dir checkpoints/$pt -filename ../CNN_DailyMail/test.json
python main_infopop.py -batch_size 1 -device 0 -predict -startind 8000 -endind 9000 -load_dir checkpoints/$pt -filename ../CNN_DailyMail/test.json
python main_infopop.py -batch_size 1 -device 0 -predict -startind 9000 -endind 10000 -load_dir checkpoints/$pt -filename ../CNN_DailyMail/test.json
python main_infopop.py -batch_size 1 -device 0 -predict -startind 10000 -endind 11000 -load_dir checkpoints/$pt -filename ../CNN_DailyMail/test.json
python main_infopop.py -batch_size 1 -device 0 -predict -startind 11000 -endind 12000 -load_dir checkpoints/$pt -filename ../CNN_DailyMail/test.json
python main_infopop.py -batch_size 1 -device 0 -predict -startind 12000 -endind 13000 -load_dir checkpoints/$pt -filename ../CNN_DailyMail/test.json
