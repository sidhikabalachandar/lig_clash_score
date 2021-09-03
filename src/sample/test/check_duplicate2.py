"""
The purpose of this code is to create the cumulative frequency and bar graphs

It can be run on sherlock using
$ $SCHRODINGER/run python3 check_duplicate2.py /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
"""

import argparse
import os
import pandas as pd
import schrodinger.structure as structure
import random
import numpy as np
import matplotlib.pyplot as plt
import schrodinger.structutils.interactions.steric_clash as steric_clash

import sys
sys.path.insert(1, '../util')
from util import *
from prot_util import *
from schrod_replacement_util import *
from lig_util import *
sys.path.insert(1, '../../../../physics_scoring')
from score_np import *
from read_vdw_params import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('--protein', type=str, default='', help='protein name')
    parser.add_argument('--target', type=str, default='', help='target ligand name')
    parser.add_argument('--start', type=str, default='', help='start ligand name')
    parser.add_argument('--max_num_correct', type=int, default=390, help='maximum number of poses considered')
    parser.add_argument('--max_num_poses_considered', type=int, default=3900, help='maximum number of poses considered')
    parser.add_argument('--num_conformers', type=int, default=300, help='maximum number of conformers considered')
    parser.add_argument('--rotation_search_step_size', type=int, default=5, help='step size between each angle '
                                                                                 'checked, in degrees')
    parser.add_argument('--grid_size', type=int, default=1, help='grid size in positive and negative x, y, z '
                                                                 'directions')
    parser.add_argument('--target_clash_cutoff', type=int, default=20, help='clash cutoff between target protein and '
                                                                            'ligand pose')
    parser.add_argument('--intolerable_cutoff', type=int, default=0, help='cutoff of max num intolerable residues')
    parser.add_argument('--index', type=int, default=-1, help='index of pose file')
    parser.add_argument('--n', type=int, default=90, help='number of files processed in each job')
    parser.add_argument('--residue_cutoff', type=int, default=3, help='name of pose group subdir')
    parser.add_argument('--rmsd_cutoff', type=float, default=2.5, help='name of pose group subdir')
    parser.add_argument('--start_clash_cutoff', type=int, default=1, help='clash cutoff between start protein and '
                                                                          'ligand pose')
    args = parser.parse_args()

    random.seed(0)

    protein, target, start = ('P00523', '4ybk', '2oiq')
    pair = '{}-to-{}'.format(target, start)
    protein_path = os.path.join(args.raw_root, protein)
    pair_path = os.path.join(protein_path, pair)
    grid_size = get_grid_size(pair_path, target, start)
    pose_path = os.path.join(pair_path, 'test_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size))

    clash_path = os.path.join(pose_path, 'clash_data')
    if not os.path.exists(clash_path):
        os.mkdir(clash_path)

    file = 'exhaustive_search_poses_4_29.csv'
    # get indices
    all_df = pd.read_csv(os.path.join(pose_path, file))
    df = all_df[all_df['start_clash'] < args.start_clash_cutoff]
    correct_df = df[df['rmsd'] <= args.rmsd_cutoff]

    incorrect_df = df[df['rmsd'] > args.rmsd_cutoff]
    incorrect_names = incorrect_df['name'].to_list()
    random.shuffle(incorrect_names)
    incorrect_names = incorrect_names[:300]
    subset_incorrect_df = incorrect_df.loc[incorrect_df['name'].isin(incorrect_names)]

    subset_df = pd.concat([correct_df, subset_incorrect_df])

    print(len(correct_df[correct_df['name'] == '292_-2,-6,-6_200,340,220']))

    print(len(incorrect_df[incorrect_df['name'] == '292_-2,-6,-6_200,340,220']))

    print(len(df[df['name'] == '292_-2,-6,-6_200,340,220']))


if __name__=="__main__":
    main()