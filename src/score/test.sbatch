#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --begin=now
#SBATCH --output=/home/users/sidhikab/lig_clash_score/src/score/run/test.out
#SBATCH --error=/home/users/sidhikab/lig_clash_score/src/score/run/test.err
#SBATCH --partition=rondror
#SBATCH --gres gpu:1

echo 'run now2!'