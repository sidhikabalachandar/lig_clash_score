"""
The purpose of this code is to create the split files
It can be run on sherlock using
$ /home/groups/rondror/software/sidhikab/miniconda/envs/geometric/bin/python split_file.py /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt
"""

import argparse
import os
from tqdm import tqdm
import pickle

graph_root = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/graphs'
data_path = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d'
PROT_CUTOFF = 405
TRAIN_CUTOFF = 0.7
TEST_CUTOFF = 0.15

def get_prots(fname):
    prots = {}
    num_codes = 0
    num_pairs = 0
    with open(fname) as fp:
        for line in tqdm(fp, desc='files'):
            if line[0] == '#': continue
            if num_pairs >= PROT_CUTOFF:
                break
            protein, target, start = line.strip().split()
            num_pairs += 1
            if protein not in prots:
                prots[protein] = []
            graph_dir = '{}/{}-to-{}_graph.pkl'.format(graph_root, target, start)
            infile = open(graph_dir, 'rb')
            graph_data = pickle.load(infile)
            infile.close()
            for _ in graph_data:
                prots[protein].append(str(num_codes) + '\n')
                num_codes += 1

    return prots, num_codes - 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('--split', type=str, default='random', help='file listing proteins to process')
    args = parser.parse_args()

    prots, num_data = get_prots(args.prot_file)
    sorted_prots = sorted(prots.items())

    train_indices = []
    val_indices = []
    test_indices = []
    for prot, indices in sorted_prots:
        if len(train_indices) / num_data <= TRAIN_CUTOFF:
            train_indices.extend(indices)
        elif len(test_indices) / num_data <= TEST_CUTOFF:
            test_indices.extend(indices)
        else:
            val_indices.extend(indices)

    print('train split', len(train_indices) / num_data)
    print('val split', len(val_indices) / num_data)
    print('test split', len(test_indices) / num_data)

    split_dir = os.path.join(data_path, 'splits')
    if not os.path.exists(split_dir):
        os.mkdir(split_dir)
    train_split = os.path.join(split_dir, f'train_{args.split}.txt')
    val_split = os.path.join(split_dir, f'val_{args.split}.txt')
    test_split = os.path.join(split_dir, f'test_{args.split}.txt')

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
