"""
The purpose of this code is to create conformers

It can be run on sherlock using
$ $SCHRODINGER/run python3 clash_graph.py /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits/search_test_incorrect_glide_index.txt /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw --group_name exhaustive_grid_1_rotation_0_360_10
"""

import argparse
import os
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_prots(docked_prot_file):
    """
    gets list of all protein, target ligands, and starting ligands in the index file
    :param docked_prot_file: (string) file listing proteins to process
    :return: process (list) list of all protein, target ligands, and starting ligands to process
    """
    process = []
    with open(docked_prot_file) as fp:
        for line in fp:
            if line[0] == '#':
                continue
            protein, target, start = line.strip().split()
            process.append((protein, target, start))

    return process


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('raw_root', type=str, help='directory where script and output files will be written')
    parser.add_argument('--group_name', type=str, default='', help='name of pose group subdir')
    args = parser.parse_args()
    random.seed(0)

    pairs = get_prots(args.docked_prot_file)
    random.shuffle(pairs)

    # for protein, target, start in pairs[:5]:
    protein = 'P02829'
    target = '2fxs'
    start = '2weq'
    pair = '{}-to-{}'.format(target, start)
    protein_path = os.path.join(args.raw_root, protein)
    pair_path = os.path.join(protein_path, pair)
    pose_path = os.path.join(pair_path, args.group_name)
    correct_clash = []
    incorrect_clash = []
    for file in os.listdir(pose_path):
        prefix = 'exhaustive_search_poses'
        if file[:len(prefix)] == prefix:
            df = pd.read_csv(os.path.join(pose_path, file))
            correct_df = df[df['rmsd'] < 2]
            correct = correct_df['start_clash'].tolist()
            correct_clash.extend(correct)
            incorrect_df = df[df['rmsd'] >= 2]
            incorrect_clash.extend(incorrect_df['start_clash'].tolist()[:len(correct)])
            print(len(correct))
            if len(correct_clash) > 200:
                break

    fig, ax = plt.subplots()
    sns.distplot(incorrect_clash, hist=True, label="incorrect pose clash")
    sns.distplot(correct_clash, hist=True, label="correct pose clash")
    plt.title('Clash Distributions for custom function')
    plt.xlabel('clash volume')
    plt.ylabel('frequency')
    ax.legend()
    fig.savefig('custom_clash.png')


if __name__ == "__main__":
    main()