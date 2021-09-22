#!/bin/bash
#SBATCH --job-name=train_scoring_model
#SBATCH --begin=now
#SBATCH --partition=rondror
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_SKU:TITAN_Xp
#SBATCH --cpus-per-task=4
#SBATCH --cores-per-socket=4
#SBATCH --output=/home/users/sidhikab/lig_clash_score/src/score/run/small_train_scoring_model2.out
#SBATCH --error=/home/users/sidhikab/lig_clash_score/src/score/run/small_train_scoring_model2.err

echo 'run now!'