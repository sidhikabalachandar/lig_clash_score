"""
The purpose of this code is to process the pdb files
It can be run on sherlock using
$ /home/groups/rondror/software/sidhikab/miniconda/envs/atom3d/bin/python decoy_dict.py /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt --out_dir /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/processed
"""

# !/usr/bin/env python
# coding: utf-8


import os
import sys

sys.path.append('..')
from tqdm import tqdm
import argparse
from string import ascii_letters

def get_num(file):
    # print(file)
    # print(file.split('_')[-1].strip(ascii_letters + '.'))
    return int(file.split('_')[-1].strip(ascii_letters + '.'))

def file_count(directory, target):
    return max(get_num(file) for file in os.listdir(directory) if 'lig' in file and target in file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='directory where PDBBind is located')
    parser.add_argument('prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('--out_dir', type=str, default=os.getcwd(), help='directory to place cleaned dataset')
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise Exception('Path not found. Please enter valid path to PDBBind dataset.')

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    dup_target = []
    all_target = []
    with open(args.prot_file) as fp:
        for line in tqdm(fp, desc='files'):
            if line[0] == '#': continue
            protein, target, start = line.strip().split()
            if target not in all_target:
                all_target.append(target)
            else:
                dup_target.append(target)

    print(dup_target)
    target_dict = {}

    with open(args.prot_file) as fp:
        for line in tqdm(fp, desc='files'):
            if line[0] == '#': continue
            protein, target, start = line.strip().split()
            pair = '{}-to-{}'.format(target, start)
            if target not in dup_target:
                target_dict[target] = pair
            else:
                num_mmcif = file_count(os.path.join(args.out_dir, target), target)
                num_sdf = file_count(os.path.join(args.data_dir, '{}/{}'.format(protein, pair)), target)
                if target == '3su0':
                    print(protein, target, start, num_mmcif, num_sdf)
                if num_mmcif == num_sdf:
                    target_dict[target] = pair

    counter = 0
    with open(args.prot_file) as fp:
        for line in tqdm(fp, desc='files'):
            if line[0] == '#': continue
            protein, target, start = line.strip().split()
            if target not in target_dict:
                print(protein, target, start)
                counter += 1

    print(counter)

if __name__ == "__main__":
    main()