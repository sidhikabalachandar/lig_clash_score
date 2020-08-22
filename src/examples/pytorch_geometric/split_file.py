"""
The purpose of this code is to create the split files
It can be run on sherlock using
$ /home/groups/rondror/software/sidhikab/miniconda/envs/geometric/bin/python split_file.py regular /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt
$ /home/groups/rondror/software/sidhikab/miniconda/envs/geometric/bin/python split_file.py MAPK14 /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt
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
    prots_pdb_codes = {}
    num_codes = 0
    num_pairs = 0
    with open(fname) as fp:
        for line in tqdm(fp, desc='files'):
            if line[0] == '#': continue
            if num_pairs >= PROT_CUTOFF:
                break
            protein, target, start = line.strip().split()
            num_pairs += 1
            if protein not in prots_pdb_codes:
                prots_pdb_codes[protein] = []
            if protein not in prots:
                prots[protein] = []
            graph_dir = '{}/{}-to-{}_graph.pkl'.format(graph_root, target, start)
            infile = open(graph_dir, 'rb')
            graph_data = pickle.load(infile)
            infile.close()
            for pdb_code in graph_data:
                prots[protein].append(str(num_codes) + '\n')
                prots_pdb_codes[protein].append(pdb_code + '\n')
                num_codes += 1

    return prots, prots_pdb_codes, num_codes - 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='file listing proteins to process')
    parser.add_argument('prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('--split', type=str, default='random', help='file listing proteins to process')
    args = parser.parse_args()

    if args.task == 'regular':
        prots, codes, num_data = get_prots(args.prot_file)
        print(len(prots))
        # sorted_prots = sorted(prots.items())
        # sorted_codes = sorted(codes.items())
        # print(len(sorted_codes), len(sorted_prots))
        #
        # train_indices = []
        # train_codes = []
        # val_indices = []
        # val_codes = []
        # test_indices = []
        # test_codes = []
        # for i in range(len(sorted_prots)):
        #     prot, indices = sorted_prots[i]
        #     _, codes = sorted_codes[i]
        #     if len(train_indices) / num_data <= TRAIN_CUTOFF:
        #         train_indices.extend(indices)
        #         train_codes.extend(codes)
        #     elif len(test_indices) / num_data <= TEST_CUTOFF:
        #         test_indices.extend(indices)
        #         test_codes.extend(codes)
        #     else:
        #         val_indices.extend(indices)
        #         val_codes.extend(codes)
        #
        # print('train split', len(train_indices) / num_data)
        # print('val split', len(val_indices) / num_data)
        # print('test split', len(test_indices) / num_data)
        #
        # split_dir = os.path.join(data_path, 'splits')
        # if not os.path.exists(split_dir):
        #     os.mkdir(split_dir)
        # # train_split = os.path.join(split_dir, f'train_{args.split}.txt')
        # # val_split = os.path.join(split_dir, f'val_{args.split}.txt')
        # # test_split = os.path.join(split_dir, f'test_{args.split}.txt')
        # #
        # # train_file = open(train_split, "w")
        # # train_file.writelines(train_indices)
        # # train_file.close()
        # #
        # # val_file = open(val_split, "w")
        # # val_file.writelines(val_indices)
        # # val_file.close()
        # #
        # # test_file = open(test_split, "w")
        # # test_file.writelines(test_indices)
        # # test_file.close()
        #
        # train_split = os.path.join(split_dir, f'train_codes_{args.split}.txt')
        # val_split = os.path.join(split_dir, f'val_codes_{args.split}.txt')
        # test_split = os.path.join(split_dir, f'test_codes_{args.split}.txt')
        #
        # train_file = open(train_split, "w")
        # train_file.writelines(train_codes)
        # train_file.close()
        #
        # val_file = open(val_split, "w")
        # val_file.writelines(val_codes)
        # val_file.close()
        #
        # test_file = open(test_split, "w")
        # test_file.writelines(test_codes)
        # test_file.close()

    elif args.task == 'MAPK14':
        ligs = ['3D83', '4F9Y']
        test_indices = []
        test_codes = []
        num_codes = 216775
        for target in ligs:
            for start in ligs:
                if target != start:
                    graph_dir = '{}/{}-to-{}_graph.pkl'.format(graph_root, target.lower(), start.lower())
                    infile = open(graph_dir, 'rb')
                    graph_data = pickle.load(infile)
                    infile.close()
                    for pdb_code in graph_data:
                        test_indices.append(str(num_codes) + '\n')
                        test_codes.append(pdb_code + '\n')
                        num_codes += 1

        split_dir = os.path.join(data_path, 'splits')
        if not os.path.exists(split_dir):
            os.mkdir(split_dir)
        test_split = os.path.join(split_dir, f'test_MAPK14.txt')
        test_file = open(test_split, "w")
        test_file.writelines(test_indices)
        test_file.close()
        test_split = os.path.join(split_dir, f'test_codes_MAPK14.txt')
        test_file = open(test_split, "w")
        test_file.writelines(test_codes)
        test_file.close()

if __name__=="__main__":
    main()
