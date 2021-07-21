

"""
The purpose of this code is to create conformers

It can be run on sherlock using
$ $SCHRODINGER/run python3 clash_search.py /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits/search_test_incorrect_glide_index.txt /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
"""

import argparse
import os
import random
import pandas as pd
import sys
sys.path.insert(1, '../util')
from util import *
from prot_util import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('raw_root', type=str, help='directory where script and output files will be written')
    parser.add_argument('--residue_cutoff', type=int, default=1, help='name of pose group subdir')
    parser.add_argument('--rmsd_cutoff', type=float, default=2.5, help='name of pose group subdir')
    parser.add_argument('--start_clash_cutoff', type=int, default=1, help='clash cutoff between start protein and '
                                                                          'ligand pose')
    args = parser.parse_args()
    random.seed(0)

    pairs = get_prots(args.docked_prot_file)
    random.shuffle(pairs)

    for residue_cutoff in [1, 2]:

    print()
    print()
    print()
    print()
    print('Simple filter: start clash cutoff is {}'.format(args.start_clash_cutoff))
    print('Advanced filter: num intolerable residue cutoff is {}'.format(residue_cutoff))
    print('Correct: rmsd cutoff is {}'.format(args.rmsd_cutoff))
    print()

    for protein, target, start in pairs[:5]:
        print()
        print(protein, target, start)
        if protein == 'Q86WV6':
            continue
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(args.raw_root, protein)
        pair_path = os.path.join(protein_path, pair)

        grid_size = get_grid_size(pair_path, target, start)
        group_name = 'test_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size)
        pose_path = os.path.join(pair_path, group_name)

        num_correct = 0
        num_total = 0
        num_after_simple_filter = 0
        num_correct_after_simple_filter = 0
        for file in os.listdir(pose_path):
            prefix = 'exhaustive_search_info'
            if file[:len(prefix)] == prefix:
                df = pd.read_csv(os.path.join(pose_path, file))
                cutoff_df = df[df['start_clash_cutoff'] == args.start_clash_cutoff]
                num_total += cutoff_df['num_poses_searched'].iloc[0]
                num_correct += cutoff_df['num_correct'].iloc[0]
                num_after_simple_filter += cutoff_df['num_after_simple_filter'].iloc[0]
                num_correct_after_simple_filter += cutoff_df['num_correct_after_simple_filter'].iloc[0]

        print('Exhaustive search, num_correct: {}, num_total: {}'.format(num_correct, num_total))
        print('After simple filter, num_correct: {}, num_total: {}'.format(num_correct_after_simple_filter,
                                                                           num_after_simple_filter))

        clash_path = os.path.join(pose_path, 'clash_data')
        total_after_subsample = 0
        total_after_filter = 0
        correct_after_subsample = 0
        correct_after_filter = 0
        for file in os.listdir(clash_path):
            prefix = 'pose_pred_data'
            if file[:len(prefix)] == prefix:
                df = pd.read_csv(os.path.join(clash_path, file))
                total_after_subsample += len(df)
                correct_after_subsample += len(df[df['rmsd'] < args.rmsd_cutoff])
                filter_df = df[df['pred_num_intolerable'] < args.residue_cutoff]
                total_after_filter += len(filter_df)
                correct_after_filter += len(filter_df[filter_df['rmsd'] < args.rmsd_cutoff])

        print('After subsample, num_correct: {}, num_total: {}'.format(correct_after_subsample, total_after_subsample))
        print('After advanced filter, num_correct: {}, num_total: {}'.format(correct_after_filter, total_after_filter))


if __name__ == "__main__":
    main()

