"""
The purpose of this code is to create the split files

It can be run on sherlock using
$ /home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python split_file.py names /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits
$ /home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python split_file.py indices /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random_conformer.txt /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits --split balance_clash_large --clash_prot_file /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random_clash_conformer.txt --cutoff 0.27 --decoy_type conformer_poses

index file: refined_random.txt
used cutoff of 0.1 for small: train/val/test = 186/39/39 protein/ligand pairs
used cutoff of 0.26 for large: train/val/test = 473/100/103 protein/ligand pairs

index file: refined_random_conformer.txt
used cutoff of 0.27 for large:
    num train 464
    num val 100
    num test 100
    train split 0.6895181321410829
    val split 0.1552409339294585
    test split 0.1552409339294585
"""

import argparse
import os
import random
import pickle
from tqdm import tqdm

def get_code_names_dict(process, max_poses):
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
        if protein not in code_dict:
            code_dict[protein] = []

        code_dict[protein].append((protein, target, start, max_poses))
        num_codes += max_poses

    return code_dict, num_codes

def get_num_decoys(process, raw_root, decoy_type):
    """
    gets list of all indices and codes for each protein
    :param process: (list) list of all protein, target ligands, and starting ligands to process
    :param raw_root: (string) path to directory with data
    :param label_file: (string) file containing rmsd label information
    :return: code_dict (dict) dict of all protein : list of all indices and codes
    :return: num_codes (int) total number of codes processed
    """
    num_codes = 0

    for protein, target, start in tqdm(process, desc='going through protein, target, start groups'):
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
        graph_dir = '{}/{}-to-{}_{}_graph.pkl'.format(pair_path, target, start, decoy_type)
        infile = open(graph_dir, 'rb')
        clustered_codes = pickle.load(infile)
        infile.close()
        num_codes += len(clustered_codes)

    return num_codes

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

def run_names(prot_file, clash_prot_file, split_dir, split, max_poses, train_prop, test_prop, val_prop, cutoff):
    process = get_prots(prot_file)
    random.shuffle(process)
    code_dict, num_codes = get_code_names_dict(process, max_poses)
    num_train = 0
    num_test = 0
    num_val = 0
    train = []
    val = []
    test = []

    if split == "random":
        for protein in code_dict:
            for _, target, start, num in code_dict[protein]:
                if num_train / num_codes <= train_prop * cutoff:
                    train.append('{} {} {}\n'.format(protein, target, start))
                    num_train += num
                elif num_test / num_codes <= test_prop * cutoff:
                    test.append('{} {} {}\n'.format(protein, target, start))
                    num_test += num
                elif num_val / num_codes <= val_prop * cutoff:
                    val.append('{} {} {}\n'.format(protein, target, start))
                    num_val += num

    elif split == "balance_clash" or split == "balance_clash_large":
        clash_process = get_prots(clash_prot_file)
        num_clash_train = 0
        num_clash_val = 0
        num_clash_test = 0
        num_non_clash_train = 0
        num_non_clash_val = 0
        num_non_clash_test = 0
        num_train_pairs = 0
        num_val_pairs = 0
        num_test_pairs = 0

        for protein in code_dict:
            for _, target, start, num in code_dict[protein]:
                if (num_clash_train + num_non_clash_train) / num_codes <= train_prop * cutoff:
                    if (protein, target, start) in clash_process and \
                            num_clash_train / num_codes <= train_prop * cutoff * 0.5:
                        train.append('{} {} {}\n'.format(protein, target, start))
                        num_clash_train += num
                        num_train_pairs += 1
                    elif num_non_clash_train / num_codes <= train_prop * cutoff * 0.5:
                        train.append('{} {} {}\n'.format(protein, target, start))
                        num_non_clash_train += num
                        num_train_pairs += 1
                elif (num_clash_test + num_non_clash_test) / num_codes <= test_prop * cutoff:
                    if (protein, target, start) in clash_process and \
                            num_clash_test / num_codes <= test_prop * cutoff * 0.5:
                        test.append('{} {} {}\n'.format(protein, target, start))
                        num_clash_test += num
                        num_test_pairs += 1
                    elif num_non_clash_test / num_codes <= test_prop * cutoff * 0.5:
                        test.append('{} {} {}\n'.format(protein, target, start))
                        num_non_clash_test += num
                        num_test_pairs += 1
                elif (num_clash_val + num_non_clash_val) / num_codes <= val_prop * cutoff:
                    if (protein, target, start) in clash_process and \
                            num_clash_val / num_codes <= val_prop * cutoff * 0.5:
                        val.append('{} {} {}\n'.format(protein, target, start))
                        num_clash_val += num
                        num_val_pairs += 1
                    elif num_non_clash_val / num_codes <= val_prop * cutoff * 0.5:
                        val.append('{} {} {}\n'.format(protein, target, start))
                        num_non_clash_val += num
                        num_val_pairs += 1

        num_train = num_clash_train + num_non_clash_train
        num_val = num_clash_val + num_non_clash_val
        num_test = num_clash_test + num_non_clash_test

        if num_train != 0 and num_val != 0 and num_test != 0:
            print('num train', num_train_pairs)
            print('num val', num_val_pairs)
            print('num test', num_test_pairs)
            print('train split', num_train / (num_train + num_val + num_test))
            print('val split', num_val / (num_train + num_val + num_test))
            print('test split', num_test / (num_train + num_val + num_test))

            if not os.path.exists(split_dir):
                os.mkdir(split_dir)

            train_split = os.path.join(split_dir, f'train_index_{split}.txt')
            val_split = os.path.join(split_dir, f'val_index_{split}.txt')
            test_split = os.path.join(split_dir, f'test_index_{split}.txt')
            combined = os.path.join(split_dir, f'combined_index_{split}.txt')

            train_file = open(train_split, "w")
            train_file.writelines(train)
            train_file.close()

            val_file = open(val_split, "w")
            val_file.writelines(val)
            val_file.close()

            test_file = open(test_split, "w")
            test_file.writelines(test)
            test_file.close()

            combined_file = open(combined, "w")
            combined_file.writelines(train)
            combined_file.writelines(val)
            combined_file.writelines(test)
            combined_file.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either names or indices')
    parser.add_argument('prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('raw_root', type=str, help='directory where raw data can be found')
    parser.add_argument('split_dir', type=str, help='path to directory where split files will be saved')
    parser.add_argument('--train', type=float, default=0.7, help='proportion of data in training set')
    parser.add_argument('--test', type=float, default=0.15, help='proportion of data in testing set')
    parser.add_argument('--val', type=float, default=0.15, help='proportion of data in testing set')
    parser.add_argument('--split', type=str, default='random', help='split name')
    parser.add_argument('--clash_prot_file', type=str, default=os.path.join(os.getcwd(), 'index.txt'),
                        help='for update task, name of prot file where clashing pairs will be placed')
    parser.add_argument('--cutoff', type=float, default=0.1, help='proportion of pdbbind data used')
    parser.add_argument('--max_poses', type=int, default=100, help='maximum number of poses considered')
    parser.add_argument('--decoy_type', type=str, default='ligand_poses', help='either cartesian_poses, ligand_poses, '
                                                                               'or conformer_poses')
    args = parser.parse_args()

    random.seed(0)
    assert(args.train + args.test + args.val == 1)

    if args.task == 'names':
        run_names(args.prot_file, args.clash_prot_file, args.split_dir, args.split, args.max_poses, args.train_prop,
                  args.test_prop, args.val_prop, args.cutoff)
    elif args.task == 'indices':
        train_split = os.path.join(args.split_dir, f'train_index_{args.split}.txt')
        val_split = os.path.join(args.split_dir, f'val_index_{args.split}.txt')
        test_split = os.path.join(args.split_dir, f'test_index_{args.split}.txt')
        train_process = get_prots(train_split)
        val_process = get_prots(val_split)
        test_process = get_prots(test_split)
        num_train = get_num_decoys(train_process, args.raw_root, args.decoy_type)
        num_val = get_num_decoys(val_process, args.raw_root, args.decoy_type)
        num_test = get_num_decoys(test_process, args.raw_root, args.decoy_type)

        print('num train', len(train_process))
        print('num val', len(val_process))
        print('num test', len(test_process))
        print('train split', num_train / (num_train + num_val + num_test))
        print('val split', num_val / (num_train + num_val + num_test))
        print('test split', num_test / (num_train + num_val + num_test))

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
