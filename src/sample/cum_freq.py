"""
The purpose of this code is to create the cumulative frequency and bar graphs

It can be run on sherlock using
/home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python cum_freq.py /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /home/users/sidhikab/lig_clash_score/reports/figures
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def bar_graph(glide_ls, score_no_vdw_ls, pose_ls, out_dir):
    fig, ax = plt.subplots()
    plt.plot(pose_ls, glide_ls, label='Glide')
    plt.plot(pose_ls, score_no_vdw_ls, label='Score no vdw')

    ax.legend()
    ax.set_xlabel('Pose Cutoff')
    ax.set_ylabel('Min RMSD')
    plt.savefig(os.path.join(out_dir, 'glide_vs_score_no_vdw_500.png'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('out_dir', type=str, help='directory where all graphs will be saved')
    parser.add_argument('--decoy_type', type=str, default='grid_search_poses', help='either cartesian_poses, '
                                                                                    'ligand_poses, or conformer_poses')
    parser.add_argument('--max_poses', type=int, default=100, help='maximum number of glide poses considered')
    args = parser.parse_args()

    protein = 'C8B467'
    target = '5jfu'
    start = '5jfp'

    pair = '{}-to-{}'.format(target, start)
    protein_path = os.path.join(args.raw_root, protein)
    pair_path = os.path.join(protein_path, pair)
    pose_path = os.path.join(pair_path, args.decoy_type)

    df = pd.read_csv(os.path.join(pose_path, 'combined.csv'))

    glide_scores = df['glide_score'].tolist()
    score_no_vdws = df['modified_score_no_vdws'].tolist()
    rmsds = df['rmsd'].tolist()
    names = df['name']
    glide_data = [(glide_scores[i], rmsds[i], names[i]) for i in range(len(rmsds))]
    score_no_vdw_data = [(score_no_vdws[i], rmsds[i], names[i]) for i in range(len(rmsds))]

    # sort data in reverse rmsd order (make sure that we choose worst in tie breakers)
    rev_glide_data = sorted(glide_data, key=lambda x: x[1], reverse=True)
    rev_score_no_vdw_data = sorted(score_no_vdw_data, key=lambda x: x[1], reverse=True)

    sorted_glide = sorted(rev_glide_data, key=lambda x: x[0])
    sorted_score_no_vdw = sorted(rev_score_no_vdw_data, key=lambda x: x[0])

    glide_ls = []
    score_no_vdw_ls = []
    pose_ls = [i for i in range(1, 500)]

    for i in range(1, 500):
        glide_ls.append(min(sorted_glide[:i], key=lambda x: x[1])[1])
        score_no_vdw_ls.append(min(sorted_score_no_vdw[:i], key=lambda x: x[1])[1])
    bar_graph(glide_ls, score_no_vdw_ls, pose_ls, args.out_dir)

if __name__=="__main__":
    main()