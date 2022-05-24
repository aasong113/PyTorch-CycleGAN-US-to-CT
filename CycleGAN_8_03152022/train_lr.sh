#!/usr/bin/env bash
#SBATCH --job-name US_syn_
#SBATCH --time=48:0:0

#SBATCH --gres gpu:1
#SBATCH --partition=gpu
#SBATCH --nodelist istar-virgo1
#SBATCH --output=/home/asong18/US_synthesis/GAN_exp3_canine_03022022/CycleGAN_shuffle=False_03022022/"%x.out%j"
#SBATCH --error=/home/asong18/US_synthesis/GAN_exp3_canine_03022022/CycleGAN_shuffle=False_03022022/"%x.err%j"

work_path=/home/asong18/US_synthesis/GAN_exp4_learning_rate_array_03082022/learning_rates_03142022/


# activate environment
source ~/anaconda2/etc/profile.d/conda.sh

# run training
python ${work_path}/train.py