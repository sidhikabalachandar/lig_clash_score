#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --begin=now
#SBATCH --output=/oak/stanford/groups/rondror/projects/ligand-docking/fragment_building/logs/EXP44/binding-affinity-%j.out
#SBATCH --error=/oak/stanford/groups/rondror/projects/ligand-docking/fragment_building/logs/EXP44/binding-affinity-%j.err
#SBATCH --partition=rondror
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-socket=2
#SBATCH --job-name=EXP44_Ares_E3nn
#SBATCH --open-mode=append
#SBATCH --ntasks=1
#SBATCH --constraint=GPU_SKU:TITAN_Xp
#SBATCH --cores-per-socket=4

eval "$(/oak/stanford/groups/rondror/projects/ligand-docking/fragment_building/software/anaconda3/bin/conda shell.bash hook)"
conda deactivate
conda deactivate
module load cuda/10.1.168
module load gcc/8.1.0
conda activate e3nn
echo 'run now!'

sh train_scoriing_model.sh