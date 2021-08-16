"""
The purpose of this code is to create the cumulative frequency and bar graphs

It can be run on sherlock using
$ $SCHRODINGER/run python3 np_score_cum_freq.py /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits/search_test_incorrect_glide_index.txt /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /home/users/sidhikab/lig_clash_score/reports/figures --protein P02829 --target 2weq --start 2yge
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


def bar_graph(glide_ls, score_no_vdw_ls, random_ls, pose_ls, args):
    fig, ax = plt.subplots()
    plt.plot(pose_ls, glide_ls, label='Glide')
    plt.plot(pose_ls, score_no_vdw_ls, label='Score no vdw')
    plt.plot(pose_ls, random_ls, label='Random')

    ax.legend()
    ax.set_xlabel('Pose Cutoff')
    ax.set_ylabel('Min RMSD')
    plt.title('Min RMSD Pose for {}_{}-to-{} using np score'.format(args.protein, args.target, args.start))
    plt.savefig(os.path.join(args.out_dir, 'np_score_{}_{}_{}.png'.format(args.protein, args.target, args.start)))


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
    print(random_datas_np.shape)
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
    args = parser.parse_args()

    random.seed(0)

    pair = '{}-to-{}'.format(args.target, args.start)
    protein_path = os.path.join(args.raw_root, args.protein)
    pair_path = os.path.join(protein_path, pair)

    grid_size = get_grid_size(pair_path, args.target, args.start)
    group_name = 'test_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size)
    pose_path = os.path.join(pair_path, group_name)

    df = pd.read_csv(os.path.join(pose_path, 'poses_after_advanced_filter.csv'))

    glide_scores = df['glide_score'].tolist()
    rmsds = df['rmsd'].tolist()
    names = df['name'].tolist()
    score_no_vdws = df['np_score_no_vdw'].tolist()

    glide_df = pd.read_csv(os.path.join(pose_path, 'glide_poses.csv'))
    glide_scores.extend(glide_df['glide_score'].tolist())
    rmsds.extend(glide_df['rmsd'].tolist())
    names.extend(glide_df['target'].tolist())
    score_no_vdws.extend(glide_df['np_score_no_vdw'].tolist())

    glide_data = [(glide_scores[i], rmsds[i], names[i]) for i in range(len(rmsds))]
    score_no_vdw_data = [(score_no_vdws[i], rmsds[i], names[i]) for i in range(len(rmsds))]
    data = [(rmsds[i], names[i]) for i in range(len(rmsds))]

    # sort data in reverse rmsd order (make sure that we choose worst in tie breakers)
    rev_glide_data = sorted(glide_data, key=lambda x: x[1], reverse=True)
    rev_score_no_vdw_data = sorted(score_no_vdw_data, key=lambda x: x[1], reverse=True)

    sorted_glide = sorted(rev_glide_data, key=lambda x: x[0])
    sorted_score_no_vdw = sorted(rev_score_no_vdw_data, key=lambda x: x[0])

    glide_ls = []
    score_no_vdw_ls = []
    pose_ls = [i for i in range(1, args.num_poses_graphed)]

    for i in range(1, args.num_poses_graphed):
        print(i)
        print("Glide:", min(sorted_glide[:i], key=lambda x: x[1]))
        print("score_no_vdw:", min(sorted_score_no_vdw[:i], key=lambda x: x[1]))
        glide_ls.append(min(sorted_glide[:i], key=lambda x: x[1])[1])
        score_no_vdw_ls.append(min(sorted_score_no_vdw[:i], key=lambda x: x[1])[1])

    random_ls = get_random_data(data, args)

    bar_graph(glide_ls, score_no_vdw_ls, random_ls, pose_ls, args)


if __name__=="__main__":
    main()