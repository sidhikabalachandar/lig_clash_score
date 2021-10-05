#!/bin/bash
#SBATCH --job-name=train_scoring_model
#SBATCH --begin=now
#SBATCH --partition=rondror
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_SKU:TITAN_Xp
#SBATCH --cpus-per-task=4
#SBATCH --cores-per-socket=4
#SBATCH --output=/home/users/sidhikab/lig_clash_score/src/score/run/balanced_train_scoring_model.out
#SBATCH --error=/home/users/sidhikab/lig_clash_score/src/score/run/balanced_train_scoring_model.err

eval "$(/oak/stanford/groups/rondror/projects/ligand-docking/fragment_building/software/anaconda3/bin/conda shell.bash hook)"
conda deactivate
conda deactivate
module load cuda/10.1.168
module load gcc/8.1.0
conda activate e3nn
echo 'run now!'
pwd

sh train_scoring_model.sh