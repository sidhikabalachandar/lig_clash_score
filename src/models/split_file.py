"""
The purpose of this code is to create the split files

It can be run on sherlock using
$ /home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python split_file.py /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/combined.csv /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits
"""

import argparse
import os
import random
import pickle
import pandas as pd
from tqdm import tqdm

CUTOFF = 0.1

def get_code_dict(process, raw_root, label_file):
    """
    gets list of all indices and codes for each protein
    :param process: (list) list of all protein, target ligands, and starting ligands to process
    :param raw_root: (string) path to directory with data
    :param label_file: (string) file containing rmsd label information
    :return: code_dict (dict) dict of all protein : list of all indices and codes
    :return: num_codes (int) total number of codes processed
    """
    code_dict = {}
    num_codes = 0
    label_df = pd.read_csv(label_file)

    for protein, target, start in tqdm(process, desc='going through protein, target, start groups'):
        if num_codes / len(label_df) > CUTOFF:
            break
        indices = []
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
        graph_dir = '{}/{}-to-{}_graph.pkl'.format(pair_path, target, start)
        infile = open(graph_dir, 'rb')
        graph_data = pickle.load(infile)
        infile.close()
        for _ in graph_data:
            indices.append(str(num_codes) + '\n')
            num_codes += 1

        if protein not in code_dict:
            code_dict[protein] = []
        code_dict[protein].extend(indices)

    return code_dict, num_codes

def get_prots(docked_prot_file):
    """
    gets list of all protein, target ligands, and starting ligands in the index file
    :param docked_prot_file: (string) file listing proteins to process
    :return: process (list) list of all protein, target ligands, and starting ligands to process
    """
    process = []
    with open(docked_prot_file) as fp:
        for line in fp:
            if line[0] == '#': continue
            protein, target, start = line.strip().split()
            process.append((protein, target, start))

    return process

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('raw_root', type=str, help='directory where raw data can be found')
    parser.add_argument('label_file', type=str, help='file with rmsd labels')
    parser.add_argument('split_dir', type=str, help='path to directory where split files will be saved')
    parser.add_argument('--train', type=float, default=0.7, help='proportion of data in training set')
    parser.add_argument('--test', type=float, default=0.15, help='proportion of data in testing set')
    parser.add_argument('--split', type=str, default='random', help='split name')
    args = parser.parse_args()

    random.seed(0)
    assert(args.train + args.test <= 1)

    process = get_prots(args.docked_prot_file)
    random.shuffle(process)
    code_dict, num_codes = get_code_dict(process, args.raw_root, args.label_file)
    prots = list(code_dict.keys())
    random.shuffle(prots)

    train_indices = []
    val_indices = []
    test_indices = []
    for protein in prots:
        indices = code_dict[protein]
        if len(train_indices) / num_codes <= args.train:
            train_indices.extend(indices)
        elif len(test_indices) / num_codes <= args.test:
            test_indices.extend(indices)
        else:
            val_indices.extend(indices)

    print('train split', len(train_indices) / num_codes)
    print('val split', len(val_indices) / num_codes)
    print('test split', len(test_indices) / num_codes)

    if not os.path.exists(args.split_dir):
        os.mkdir(args.split_dir)
    train_split = os.path.join(args.split_dir, f'train_{args.split}.txt')
    val_split = os.path.join(args.split_dir, f'val_{args.split}.txt')
    test_split = os.path.join(args.split_dir, f'test_{args.split}.txt')

    train_file = open(train_split, "w")
    train_file.writelines(train_indices)
    train_file.close()

    val_file = open(val_split, "w")
    val_file.writelines(val_indices)
    val_file.close()

    test_file = open(test_split, "w")
    test_file.writelines(test_indices)
    test_file.close()

if __name__=="__main__":
    main()
