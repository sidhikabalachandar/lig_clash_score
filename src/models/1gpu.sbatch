#!/bin/bash
#SBATCH --partition rondror
#SBATCH --output /home/users/sidhikab/lig_clash_score/src/models/out/v2.out
#SBATCH --mail-type=FAIL
#SBATCH --ntasks-per-socket=1
#SBATCH --gres-flags=enforce-binding
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=96:00:00
source /home/users/sidhikab/lig_clash_score/src/models/.geometric
$@
