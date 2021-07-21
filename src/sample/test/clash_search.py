

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
    parser.add_argument('--residue_cutoff', type=int, default=0, help='name of pose group subdir')
    parser.add_argument('--rmsd_cutoff', type=float, default=2, help='name of pose group subdir')
    args = parser.parse_args()
    random.seed(0)

    pairs = get_prots(args.docked_prot_file)
    random.shuffle(pairs)

    for protein, target, start in pairs[:5]:
        print(protein, target, start)
        if protein == 'Q86WV6':
            continue
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(args.raw_root, protein)
        pair_path = os.path.join(protein_path, pair)

        grid_size = get_grid_size(pair_path, target, start)
        group_name = 'exhaustive_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size)
        pose_path = os.path.join(pair_path, group_name)

        num_correct = 0
        num_total = 0
        num_after_simple_filter = 0
        num_correct_after_simple_filter = 0
        for file in os.listdir(pose_path):
            prefix = 'exhaustive_search_info'
            if file[:len(prefix)] == prefix:
                df = pd.read_csv(os.path.join(pose_path, file))
                num_total += df['num_poses_searched'].iloc[0]
                num_correct += df['num_correct'].iloc[0]
                num_after_simple_filter += df['num_after_simple_filter'].iloc[0]
                num_correct_after_simple_filter += df['num_correct_after_simple_filter'].iloc[0]

        print('Before simple filter, num_correct: {}, num_total: {}'.format(num_correct, num_total))
        print('After simple filter, num_correct: {}, num_total: {}'.format(num_correct_after_simple_filter,
                                                                           num_after_simple_filter))

        # correct_path = os.path.join(pose_path, 'correct_after_simple_filter')
        # clash_path = os.path.join(correct_path, 'clash_data')
        # file = os.path.join(clash_path, 'combined.csv')
        # df = pd.read_csv(file)
        # num_correct = len(df[df['pred_num_intolerable'] <= args.residue_cutoff])
        # print('After advanced filter for correct poses: num_before_filter: {}, num_after_filter: {}'.format(len(df),
        #                                                                                                     num_correct))

        # subsample_path = os.path.join(pose_path, 'subsample_incorrect')
        # clash_path = os.path.join(subsample_path, 'clash_data')
        # total_before_filter = 0
        # total_after_filter = 0
        # for file in os.listdir(clash_path):
        #     prefix = 'pose_pred_data'
        #     if file[:len(prefix)] == prefix:
        #         df = pd.read_csv(os.path.join(clash_path, file))
        #         total_before_filter += len(df)
        #         df = df[df['pred_num_intolerable'] <= args.residue_cutoff]
        #         total_after_filter += len(df)
        #
        # print('Before advanced filter for incorrect poses, num_total: {}'.format(total_before_filter))
        # print('After advanced filter for incorrect poses, num_total: {}'.format(total_after_filter))


if __name__ == "__main__":
    main()

