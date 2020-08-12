#!/bin/bash
#SBATCH --partition rondror
#SBATCH --output /home/users/sidhikab/lig_clash_score/src/examples/pytorch_geometric/out/train_pdbbind.out
#SBATCH --mail-type=FAIL
#SBATCH --ntasks-per-socket=1
#SBATCH --gres-flags=enforce-binding
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=48:00:00
source /home/users/sidhikab/lig_clash_score/src/examples/pytorch_geometric/.geometric
$@
