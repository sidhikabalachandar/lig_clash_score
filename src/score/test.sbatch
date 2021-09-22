#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --begin=now
#SBATCH --output=/home/users/sidhikab/lig_clash_score/src/score/run/small_train_scoring_model.out
#SBATCH --error=/home/users/sidhikab/lig_clash_score/src/score/run/small_train_scoring_model.err
#SBATCH --partition=rondror
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-socket=2
#SBATCH --job-name=train_scoring_model
#SBATCH --open-mode=append
#SBATCH --ntasks=1
#SBATCH --constraint=GPU_SKU:TITAN_Xp
#SBATCH --cores-per-socket=4

echo 'run now!'