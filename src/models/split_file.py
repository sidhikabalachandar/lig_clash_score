"""
The purpose of this code is to create the split files

It can be run on sherlock using
$ /home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python split_file.py /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits
$ /home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python split_file.py /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits --split balance_clash --clash_prot_file /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random_clash.txt
"""

import argparse
import os
import random
import pickle
import pandas as pd
from tqdm import tqdm

def get_code_dict(process, raw_root):
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

    for protein, target, start in tqdm(process, desc='going through protein, target, start groups'):
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
        graph_dir = '{}/{}-to-{}_clustered.pkl'.format(pair_path, target, start)
        infile = open(graph_dir, 'rb')
        clustered_codes = pickle.load(infile)
        infile.close()
        if protein not in code_dict:
            code_dict[protein] = []
        code_dict[protein].append((protein, target, start, len(clustered_codes)))
        num_codes += len(clustered_codes)

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
    parser.add_argument('split_dir', type=str, help='path to directory where split files will be saved')
    parser.add_argument('--train', type=float, default=0.7, help='proportion of data in training set')
    parser.add_argument('--test', type=float, default=0.15, help='proportion of data in testing set')
    parser.add_argument('--val', type=float, default=0.15, help='proportion of data in testing set')
    parser.add_argument('--split', type=str, default='random', help='split name')
    parser.add_argument('--clash_prot_file', type=str, default=os.path.join(os.getcwd(), 'index.txt'),
                        help='for update task, name of prot file where clashing pairs will be placed')
    parser.add_argument('--cutoff', type=float, default=0.1, help='proportion of pdbbind data used')
    args = parser.parse_args()

    random.seed(0)
    assert(args.train + args.test + args.val == 1)

    process = get_prots(args.docked_prot_file)
    random.shuffle(process)
    code_dict, num_codes = get_code_dict(process, args.raw_root)
    num_train = 0
    num_test = 0
    num_val = 0
    train = []
    val = []
    test = []

    if args.split == "random":
        for protein in code_dict:
            for _, target, start, num in code_dict[protein]:
                if num_train / num_codes <= args.train * args.cutoff:
                    train.append('{} {} {}\n'.format(protein, target, start))
                    num_train += num
                elif num_test / num_codes <= args.test * args.cutoff:
                    test.append('{} {} {}\n'.format(protein, target, start))
                    num_test += num
                elif num_val / num_codes <= args.val * args.cutoff:
                    val.append('{} {} {}\n'.format(protein, target, start))
                    num_val += num

    elif args.split == "balance_clash":
        clash_process = get_prots(args.clash_prot_file)
        num_clash_train = 0
        num_clash_val = 0
        num_clash_test = 0
        num_non_clash_train = 0
        num_non_clash_val = 0
        num_non_clash_test = 0

        for protein in code_dict:
            for _, target, start, num in code_dict[protein]:
                if (num_clash_train + num_non_clash_train) / num_codes <= args.train * args.cutoff:
                    if (protein, target, start) in clash_process and \
                            num_clash_train / num_codes <= args.train * args.cutoff * 0.5:
                        train.append('{} {} {}\n'.format(protein, target, start))
                        num_clash_train += num
                    elif num_non_clash_train / num_codes <= args.train * args.cutoff * 0.5:
                        train.append('{} {} {}\n'.format(protein, target, start))
                        num_non_clash_train += num
                elif (num_clash_test + num_non_clash_test) / num_codes <= args.test * args.cutoff:
                    if (protein, target, start) in clash_process and \
                            num_clash_test / num_codes <= args.test * args.cutoff * 0.5:
                        test.append('{} {} {}\n'.format(protein, target, start))
                        num_clash_test += num
                    elif num_non_clash_test / num_codes <= args.test * args.cutoff * 0.5:
                        test.append('{} {} {}\n'.format(protein, target, start))
                        num_non_clash_test += num
                elif (num_clash_val + num_non_clash_val) / num_codes <= args.val * args.cutoff:
                    if (protein, target, start) in clash_process and \
                            num_clash_val / num_codes <= args.val * args.cutoff * 0.5:
                        val.append('{} {} {}\n'.format(protein, target, start))
                        num_clash_val += num
                    elif num_non_clash_val / num_codes <= args.val * args.cutoff * 0.5:
                        val.append('{} {} {}\n'.format(protein, target, start))
                        num_non_clash_val += num

        num_train = num_clash_train + num_non_clash_train
        num_val = num_clash_val + num_non_clash_val
        num_test = num_clash_test + num_non_clash_test

    if num_train != 0 and num_val != 0 and num_test != 0:
        print('train split', num_train / (num_train + num_val + num_test))
        print('val split', num_val / (num_train + num_val + num_test))
        print('test split', num_test / (num_train + num_val + num_test))

        if not os.path.exists(args.split_dir):
            os.mkdir(args.split_dir)

        train_split = os.path.join(args.split_dir, f'train_index_{args.split}.txt')
        val_split = os.path.join(args.split_dir, f'val_index_{args.split}.txt')
        test_split = os.path.join(args.split_dir, f'test_index_{args.split}.txt')

        train_file = open(train_split, "w")
        train_file.writelines(train)
        train_file.close()

        val_file = open(val_split, "w")
        val_file.writelines(val)
        val_file.close()

        test_file = open(test_split, "w")
        test_file.writelines(test)
        test_file.close()

        train_split = os.path.join(args.split_dir, f'train_{args.split}.txt')
        val_split = os.path.join(args.split_dir, f'val_{args.split}.txt')
        test_split = os.path.join(args.split_dir, f'test_{args.split}.txt')
        train_indices = [str(i) + '\n' for i in range(num_train)]
        val_indices = [str(i) + '\n' for i in range(num_train, num_train + num_val)]
        test_indices = [str(i) + '\n' for i in range(num_train + num_val, num_train + num_val + num_test)]

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
