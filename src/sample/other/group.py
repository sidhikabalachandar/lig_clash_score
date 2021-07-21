"""
The purpose of this code is to create conformers

It can be run on sherlock using
$ $SCHRODINGER/run python3 group.py delete /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits/search_test_incorrect_glide_index.txt /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw --protein P00797 --target 3own --start 3d91 --index 0 --n 1
"""

import argparse
import random
import os
import pandas as pd
import pickle
import sys
sys.path.insert(1, '../util')
from util import *
from prot_util import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either align or search')
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('raw_root', type=str, help='directory where script and output files will be written')
    parser.add_argument('--rmsd_cutoff', type=int, default=2.5,
                        help='rmsd accuracy cutoff between predicted ligand pose '
                             'and true ligand pose')
    args = parser.parse_args()
    random.seed(0)

    if args.task == 'run':
        pairs = get_prots(args.docked_prot_file)
        random.shuffle(pairs)
        for protein, target, start in pairs[:5]:
            if protein == 'Q86WV6':
                continue

            pair = '{}-to-{}'.format(target, start)
            protein_path = os.path.join(args.raw_root, protein)
            pair_path = os.path.join(protein_path, pair)
            grid_size = get_grid_size(pair_path, target, start)
            pose_path = os.path.join(pair_path, 'test_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size))
            correct_path = os.path.join(pose_path, 'correct_after_simple_filter')
            if not os.path.exists(correct_path):
                os.mkdir(correct_path)
            subsample_path = os.path.join(pose_path, 'subsample_incorrect_after_simple_filter')
            if not os.path.exists(subsample_path):
                os.mkdir(subsample_path)

            prefix = 'exhaustive_search_poses_'
            suffix = '.csv'
            files = [f for f in os.listdir(pose_path) if f[:len(prefix)] == prefix]

            for file in files:
                name = file[len(prefix):-len(suffix)]
                df = pd.read_csv(os.path.join(pose_path, file))
                correct_df = df[df['rmsd'] <= args.rmsd_cutoff]
                correct_indices = correct_df.index
                outfile = open(os.path.join(subsample_path, 'index_{}.pkl'.format(name)), 'wb')
                pickle.dump(correct_indices, outfile)

                incorrect_df = df[df['rmsd'] > args.rmsd_cutoff]
                incorrect_indices = [i for i in incorrect_df.index]
                random.shuffle(incorrect_indices)
                incorrect_indices = incorrect_indices[:300]
                incorrect_indices = sorted(incorrect_indices)
                outfile = open(os.path.join(subsample_path, 'index_{}.pkl'.format(name)), 'wb')
                pickle.dump(incorrect_indices, outfile)
    elif args.task == "delete":
        pairs = get_prots(args.docked_prot_file)
        random.shuffle(pairs)
        for protein, target, start in pairs[:5]:
            if protein == 'Q86WV6':
                continue

            pair = '{}-to-{}'.format(target, start)
            protein_path = os.path.join(args.raw_root, protein)
            pair_path = os.path.join(protein_path, pair)
            grid_size = get_grid_size(pair_path, target, start)
            pose_path = os.path.join(pair_path, 'test_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size))
            correct_path = os.path.join(pose_path, 'correct_after_simple_filter')
            subsample_path = os.path.join(pose_path, 'subsample_incorrect_after_simple_filter')
            os.system('rm -rf {}'.format(correct_path))
            os.system('rm -rf {}'.format(subsample_path))


if __name__ == "__main__":
    main()
