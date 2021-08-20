"""
The purpose of this code is to create the cumulative frequency and bar graphs

It can be run on sherlock using
$ $SCHRODINGER/run python3 compare_rank.py /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits/search_test_incorrect_glide_index.txt /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /home/users/sidhikab/lig_clash_score/reports/figures --protein P02829 --target 2weq --start 2yge
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np

import sys
sys.path.insert(1, '../util')
from util import *
from prot_util import *
from lig_util import *


def get_random_data(data, args):
    random_datas = []
    for _ in range(100):
        random.shuffle(data)
        random_data = [(i, data[i][0], data[i][1]) for i in range(len(data))]
        rev_random_data = sorted(random_data, key=lambda x: x[1], reverse=True)
        sorted_random = sorted(rev_random_data, key=lambda x: x[0])
        random_ls = []

        for i in range(1, args.num_poses_graphed):
            random_ls.append(min(sorted_random[:i], key=lambda x: x[1])[1])

        random_datas.append(random_ls)

    random_datas_np = np.array(random_datas)
    random_datas_np = np.mean(random_datas_np, axis=0)
    return random_datas_np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('out_dir', type=str, help='directory where all graphs will be saved')
    parser.add_argument('--protein', type=str, default='', help='protein name')
    parser.add_argument('--target', type=str, default='', help='target ligand name')
    parser.add_argument('--start', type=str, default='', help='start ligand name')
    parser.add_argument('--max_poses', type=int, default=100, help='maximum number of glide poses considered')
    parser.add_argument('--num_conformers', type=int, default=300, help='maximum number of conformers considered')
    parser.add_argument('--rotation_search_step_size', type=int, default=5, help='step size between each angle '
                                                                                 'checked, in degrees')
    parser.add_argument('--grid_size', type=int, default=1, help='grid size in positive and negative x, y, z '
                                                                 'directions')
    parser.add_argument('--target_clash_cutoff', type=int, default=20, help='clash cutoff between target protein and '
                                                                            'ligand pose')
    parser.add_argument('--intolerable_cutoff', type=int, default=0, help='cutoff of max num intolerable residues')
    parser.add_argument('--num_poses_graphed', type=int, default=600, help='cutoff of max num intolerable residues')
    parser.add_argument('--rmsd_cutoff', type=float, default=2.5, help='name of pose group subdir')
    args = parser.parse_args()

    random.seed(0)

    # for protein, target, start in [('P02829', '2weq', '2yge'), ('P00797', '3own', '3d91'), ('C8B467', '5ult', '5uov'),
    #                                ('P03368', '1gno', '1zp8'), ('P02829', '2fxs', '2weq'), ('P11838', '3wz6', '1gvx'),
    #                                ('P00523', '4ybk', '2oiq'), ('P00519', '4twp', '5hu9'), ('P0DOX7', '6msy', '6mub')]:

    for protein, target, start in [('P02829', '2weq', '2yge'), ('P00797', '3own', '3d91'), ('C8B467', '5ult', '5uov')]:
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(args.raw_root, protein)
        pair_path = os.path.join(protein_path, pair)

        grid_size = get_grid_size(pair_path, target, start)
        group_name = 'test_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size)
        pose_path = os.path.join(pair_path, group_name)

        df = pd.read_csv(os.path.join(pose_path, 'poses_after_advanced_filter.csv'))

        rmsds = df['rmsd'].tolist()
        names = df['name'].tolist()
        python_scores = df['python_score'].tolist()

        glide_df = pd.read_csv(os.path.join(pose_path, 'glide_poses.csv'))
        for i in range(1, 100):
            pose_df = glide_df[glide_df['target'] == '{}_lig{}'.format(target, i)]
            if len(pose_df) > 0:
                names.append(pose_df['target'].iloc[0])
                rmsds.append(pose_df['rmsd'].iloc[0])
                python_scores.append(pose_df['python_score'].iloc[0])

        python_score_data = [(python_scores[i], rmsds[i], names[i]) for i in range(len(rmsds))]
        data = [(rmsds[i], names[i]) for i in range(len(rmsds))]

        # sort data in reverse rmsd order (make sure that we choose worst in tie breakers)
        rev_python_score_data = sorted(python_score_data, key=lambda x: x[1], reverse=True)

        sorted_python_vdw = sorted(rev_python_score_data, key=lambda x: x[0])

        python_score_ls = []

        for i in range(1, args.num_poses_graphed):
            python_score_ls.append(min(sorted_python_vdw[:i], key=lambda x: x[1])[1])

        random_ls = get_random_data(data, args)

        for i in range(len(python_score_ls)):
            if python_score_ls[i] < args.rmsd_cutoff:
                python_rank = i
                break

        for i in range(len(random_ls)):
            if random_ls[i] < args.rmsd_cutoff:
                random_rank = i
                break

        print(protein, target, start)
        print('First rank at which correct pose found using python score:', python_rank)
        print('First rank at which correct pose found using random score:', random_rank)
        print()


if __name__=="__main__":
    main()