'''
This protocol can be used to find the amino acid sequence for a set of structures
The result is stored in a pickled 2D array
The 2D array will be used for pairwise alignment

Store outputs in Data/Alignments
Store 1 alignment pickled file per protein

how to run this file:
/home/groups/rondror/software/sidhikab/miniconda/envs/geometric/bin/python index_update.py /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt
'''

import argparse
import os
import pickle
from tqdm import tqdm

save_path = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    args = parser.parse_args()

    infile = open(os.path.join(save_path, 'chains.pkl'), 'rb')
    chains = pickle.load(infile)
    infile.close()

    text = []
    with open(args.docked_prot_file) as fp:
        for line in tqdm(fp, desc='protein file'):
            if line[0] == '#': continue
            protein, target, start = line.strip().split()
            if protein in chains:
                text.append(line)

    file = open(os.path.join(save_path, 'new_refined_random.txt'), "w")
    file.writelines(text)
    file.close()

if __name__ == '__main__':
    main()