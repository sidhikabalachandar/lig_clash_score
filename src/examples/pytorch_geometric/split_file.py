"""
The purpose of this code is to create the split files
It can be run on sherlock using
$ /home/groups/rondror/software/sidhikab/miniconda/envs/atom3d/bin/python split_file.py
"""

import argparse
import os

data_path = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=float, default=0.6,  help='proportion of data used for training')
    parser.add_argument('--val', type=float, default=0.2,  help='proportion of data used for validation')
    parser.add_argument('--split', type=str, default='60_20_20', help='file name')
    parser.add_argument('--size', type=int, default=97, help='size of data')
    args = parser.parse_args()

    split_dir = os.path.join(data_path, 'splits')
    if not os.path.exists(split_dir):
        os.mkdir(split_dir)
    train_split = os.path.join(split_dir, f'train_{args.split}.txt')
    val_split = os.path.join(split_dir, f'val_{args.split}.txt')
    test_split = os.path.join(split_dir, f'test_{args.split}.txt')

    indices = [str(i) + '\n' for i in range(args.size)]

    train_file = open(train_split, "w")
    train_file.writelines(indices[:int(args.size * args.train)])
    train_file.close()

    val_file = open(val_split, "w")
    val_file.writelines(indices[int(args.size * args.train):int(args.size * (args.train + args.val))])
    val_file.close()

    test_file = open(test_split, "w")
    test_file.writelines(indices[int(args.size * (args.train + args.val)):])
    test_file.close()

if __name__=="__main__":
    main()
