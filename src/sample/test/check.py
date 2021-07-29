"""
The purpose of this code is to first create the raw directory folder and include the following files
starting protein receptor
starting ligand
target ligand
glide pose viewer file

Then the top glide poses are added

Then the decoys are created

It can be run on sherlock using
$ $SCHRODINGER/run python3 test.py /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits/search_test_incorrect_glide_index.txt /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d --protein P00797 --target 3own --start 3d91 --index 0
"""

import argparse
import os
import random
import pandas as pd
import math
import pickle
import schrodinger.structure as structure
from schrodinger.structutils.transform import get_centroid
import schrodinger.structutils.interactions.steric_clash as steric_clash
import time
import sys
sys.path.insert(1, '../util')
from util import *
from prot_util import *
from schrod_replacement_util import *


X_AXIS = [1.0, 0.0, 0.0]  # x-axis unit vector
Y_AXIS = [0.0, 1.0, 0.0]  # y-axis unit vector
Z_AXIS = [0.0, 0.0, 1.0]  # z-axis unit vector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('root', type=str, help='directory where raw data will be placed')
    parser.add_argument('--index', type=int, default=-1, help='index of pose file')
    parser.add_argument('--protein', type=str, default='', help='name of protein')
    parser.add_argument('--target', type=str, default='', help='name of target ligand')
    parser.add_argument('--start', type=str, default='', help='name of start ligand')
    parser.add_argument('--save_pred_path', type=str, help='prediction graph file')
    parser.add_argument('--save_true_path', type=str, help='true graph file')
    parser.add_argument('--target_clash_cutoff', type=int, default=0, help='clash cutoff between target protein and '
                                                                            'ligand pose')
    parser.add_argument('--intolerable_cutoff', type=int, default=0, help='cutoff of max num intolerable residues')
    parser.add_argument('--rmsd_cutoff', type=int, default=2.5,
                        help='rmsd accuracy cutoff between predicted ligand pose '
                             'and true ligand pose')
    parser.add_argument('--n', type=int, default=9, help='number of files processed in each job')
    args = parser.parse_args()
    random.seed(0)

    raw_root = os.path.join(args.root, 'raw')

    pairs = get_prots(args.docked_prot_file)
    random.shuffle(pairs)
    for protein, target, start in pairs[:5]:
        if protein == 'Q86WV6':
            continue
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, pair)
        grid_size = get_grid_size(pair_path, target, start)
        pose_path = os.path.join(pair_path, 'test_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size))
        prefix = 'exhaustive_search_poses_'
        files = [f for f in os.listdir(pose_path) if f[:len(prefix)] == prefix]
        grouped_files = group_files(args.n, files)

        counter = 0

        for file in grouped_files:
            # get indices
            df = pd.read_csv(os.path.join(pose_path, file))
            correct_df = df[df['rmsd'] <= args.rmsd_cutoff]
            correct_indices = [i for i in correct_df.index]
            counter += len(correct_indices)

        print(protein, target, start, counter)
        break


if __name__ == "__main__":
    main()
