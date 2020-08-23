"""
The purpose of this code is to set the train, val, and test data sets
It can be run on sherlock using
/home/groups/rondror/software/sidhikab/miniconda/envs/geometric/bin/python time_check.py
"""

import os
from tqdm import tqdm

CUTOFF = 1590000000.0
prot_file = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random_with_unaligned.txt'
data_root = '/oak/stanford/groups/rondror/projects/ligand-docking/pdbbind_2019/data'
save_path = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random2.txt'

def main():
    original = {}

    for protein in tqdm(os.listdir(data_root), desc='proteins in data root'):
        protein_root = os.path.join(data_root, '{}/structures/aligned'.format(protein))
        for file in os.listdir(protein_root):
            full_file = os.path.join(protein_root, file)
            if os.path.getmtime(full_file) > CUTOFF:
                if 'lig' in file or 'prot' in file:
                    os.remove(full_file)
                    # print(full_file)

        docking_root = os.path.join(data_root, '{}/docking/sp_es4'.format(protein))
        if os.path.exists(docking_root):
            for pair in os.listdir(docking_root):
                pair_root = os.path.join(docking_root, pair)
                for file in os.listdir(pair_root):
                    full_file = os.path.join(pair_root, file)
                    if os.path.getmtime(full_file) > CUTOFF:
                        os.remove(full_file)
                        # print(full_file)


if __name__=="__main__":
    main()