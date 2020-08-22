"""
The purpose of this code is to create the split files
It can be run on sherlock using
$ /home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python cum_freq.py regular /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits/test_codes_random.txt /home/users/sidhikab/lig_clash_score/models/logs/2020-08-12-15-20-55 /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/pdbbind_refined_set_labels.csv /home/users/sidhikab/lig_clash_score/reports/figures
$ /home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python cum_freq.py MAPK14 /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits/test_codes_MAPK14.txt /home/users/sidhikab/lig_clash_score/models/logs/2020-08-12-15-20-55 /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/pdbbind_refined_set_labels.csv /home/users/sidhikab/lig_clash_score/reports/figures
"""

import argparse
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import sys

sys.path[-2] = '/home/users/sidhikab/lig_clash_score/src'

from atom3d.protein_ligand.get_labels import get_label

MAX_POSES = 100
CUTOFF = 2
LABELS =['gnn', 'glide']

def graph(title, ls, save_root):
    n_bins = 1000
    fig, ax = plt.subplots()

    # plot the cumulative histogram
    for i in range(len(ls)):
        ax.hist(ls[i], n_bins, density=True, histtype='step', cumulative=True, label=LABELS[i])
    ax.grid(True)
    ax.legend(loc='lower right')
    ax.set_title(title + ' Pose Cumulative step histograms')
    ax.set_xlabel('Docking performance (RMSD)')
    ax.set_ylabel('Cumulative Frequency')

    plt.savefig(os.path.join(save_root, title + '.png'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='file listing proteins to process')
    parser.add_argument('test_split', type=str, help='file listing proteins to process')
    parser.add_argument('log_dir', type=str, help='file listing proteins to process')
    parser.add_argument('label_file', type=str, help='file listing proteins to process')
    parser.add_argument('graph_save_root', type=str, help='file listing proteins to process')
    args = parser.parse_args()

    test_codes = []
    with open(args.test_split) as f:
        for line in f:
            test_codes.append(line.strip())

    infile = open(os.path.join(args.log_dir, 'y_pred_MAPK14.pkl'), 'rb')
    y_pred = pickle.load(infile)
    infile.close()

    ligs = ['3D83', '4F9Y']
    dfs = []
    for target in ligs:
        for start in ligs:
            if target != start:
                label_file = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/labels/{}-to-{}.csv'.format(target, start)
                dfs.append(pd.read_csv(label_file))

    label_df = pd.concat(dfs)

    if args.task == 'regular':
        if os.path.exists(os.path.join(args.log_dir, 'pred_dict.pkl')):
            infile = open(os.path.join(args.log_dir, 'pred_dict.pkl'), 'rb')
            pred_dict = pickle.load(infile)
            infile.close()
        else:
            pred_dict = {}

            # build dict for gnn predictions
            for i in tqdm(range(len(test_codes)), desc='pdb codes in gnn predictions'):
                lig = test_codes[i][:4]
                if lig not in pred_dict:
                    pred_dict[lig] = []

                pred_dict[lig].append((test_codes[i], y_pred[i], get_label(test_codes[i], label_df)))

            for lig in pred_dict:
                pred_dict[lig] = sorted(pred_dict[lig], key=lambda x: x[1])

            outfile = open(os.path.join(args.log_dir, 'pred_dict.pkl'), 'wb')
            pickle.dump(pred_dict, outfile)

        if os.path.exists(os.path.join(args.log_dir, 'glide_dict.pkl')):
            infile = open(os.path.join(args.log_dir, 'glide_dict.pkl'), 'rb')
            glide_dict = pickle.load(infile)
            infile.close()
        else:
            glide_dict = {}

            # build dict for glide predictions
            for lig in tqdm(pred_dict, desc='pdb codes in glide predictions'):
                for i in range(1, MAX_POSES + 1):
                    pdb_code = '{}_lig{}'.format(lig, str(i))
                    if len(label_df[label_df['target'] == pdb_code]) != 0:
                        if lig not in glide_dict:
                            glide_dict[lig] = []
                        glide_dict[lig].append((pdb_code, get_label(pdb_code, label_df)))

            outfile = open(os.path.join(args.log_dir, 'glide_dict.pkl'), 'wb')
            pickle.dump(glide_dict, outfile)

        # create empty top_ls index 0 is gnn pred index 1 is glide
        top_ls = [[], []]
        any_ls = [[], []]
        any_ls_wo_ground_truth = [[], []]

        for lig in pred_dict:
            if lig in glide_dict and len(pred_dict[lig]) >= len(glide_dict[lig]):
                top_ls[0].append(pred_dict[lig][0][2])
                top_ls[1].append(glide_dict[lig][0][1])
                if min([pred_dict[lig][i][2] for i in range(len(glide_dict[lig]))]) == 0:
                    # print(label_df[label_df['target'] == '{}_lig0'.format(lig)]['protein'].iloc[0], lig,
                    #       label_df[label_df['target'] == '{}_lig0'.format(lig)]['start'].iloc[0],
                    #       min([pred_dict[lig][i][2] for i in range(len(glide_dict[lig]))]),
                    #       sorted([pred_dict[lig][i][1] for i in range(len(glide_dict[lig]))])[1],
                    #       sorted([glide_dict[lig][i] for i in range(len(glide_dict[lig]))], key=lambda x: x[1])[0])
                    any_ls_wo_ground_truth[0].append(
                        sorted([pred_dict[lig][i][1] for i in range(len(glide_dict[lig]))])[1])
                else:
                    any_ls_wo_ground_truth[0].append(min([pred_dict[lig][i][2] for i in range(len(glide_dict[lig]))]))

                any_ls[0].append(min([pred_dict[lig][i][2] for i in range(len(glide_dict[lig]))]))
                any_ls[1].append(min([glide_dict[lig][i][1] for i in range(len(glide_dict[lig]))]))
                any_ls_wo_ground_truth[1].append(min([glide_dict[lig][i][1] for i in range(len(glide_dict[lig]))]))

        top_freq_ls = []
        any_freq_ls = []
        any_wo_ground_truth_freq_ls = []
        for i in range(len(top_ls)):
            top_freq_ls.append(len([j for j in top_ls[i] if j < CUTOFF]) * 100 / len(top_ls[i]))
            any_freq_ls.append(len([j for j in any_ls[i] if j < CUTOFF]) * 100 / len(any_ls[i]))
            any_wo_ground_truth_freq_ls.append(len([j for j in any_ls_wo_ground_truth[i] if j < CUTOFF]) * 100 / len(any_ls_wo_ground_truth[i]))

        print("Labels:", LABELS)
        print("Top frequencies:", top_freq_ls)
        print("Any frequency:", any_freq_ls)
        print("Any without ground truth frequency:", any_wo_ground_truth_freq_ls)
        graph('top', top_ls, args.graph_save_root)
        graph('any', any_ls, args.graph_save_root)
        graph('any_without_ground_truth', any_ls_wo_ground_truth, args.graph_save_root)

    elif args.task == 'MAPK14':
        pred_dict = {}

        #build dict for gnn predictions
        for i in tqdm(range(len(test_codes)), desc='pdb codes in gnn predictions'):
            lig = test_codes[i][:4]
            if lig not in pred_dict:
                pred_dict[lig] = []

            pred_dict[lig].append((test_codes[i], y_pred[i], get_label(test_codes[i], label_df)))

        for lig in pred_dict:
            pred_dict[lig] = sorted(pred_dict[lig], key=lambda x: x[1])

        glide_dict = {}

        #build dict for glide predictions
        for lig in tqdm(pred_dict, desc='pdb codes in glide predictions'):
            for i in range(1, MAX_POSES + 1):
                pdb_code = '{}_lig{}'.format(lig, str(i))
                if len(label_df[label_df['target'] == pdb_code]) != 0:
                    if lig not in glide_dict:
                        glide_dict[lig] = []
                    glide_dict[lig].append((pdb_code, get_label(pdb_code, label_df)))

        #create empty top_ls index 0 is gnn pred index 1 is glide
        top_ls = [[], []]
        any_ls = [[], []]
        any_ls_wo_ground_truth = [[], []]
        ligs = []

        for lig in pred_dict:
            if lig in glide_dict and len(pred_dict[lig]) >= len(glide_dict[lig]):
                top_ls[0].append(pred_dict[lig][0][2])
                top_ls[1].append(glide_dict[lig][0][1])
                if min([pred_dict[lig][i][2] for i in range(len(glide_dict[lig]))]) == 0:
                    any_ls_wo_ground_truth[0].append(sorted([pred_dict[lig][i][1] for i in range(len(glide_dict[lig]))])[1])
                else:
                    any_ls_wo_ground_truth[0].append(min([pred_dict[lig][i][2] for i in range(len(glide_dict[lig]))]))

                any_ls[0].append(min([pred_dict[lig][i][2] for i in range(len(glide_dict[lig]))]))
                any_ls[1].append(min([glide_dict[lig][i][1] for i in range(len(glide_dict[lig]))]))
                any_ls_wo_ground_truth[1].append(min([glide_dict[lig][i][1] for i in range(len(glide_dict[lig]))]))

                ligs.append((lig, sorted([pred_dict[lig][i] for i in range(len(glide_dict[lig]))], key=lambda x: x[2])[0], sorted([glide_dict[lig][i] for i in range(len(glide_dict[lig]))], key=lambda x: x[1])[0]))

        sorted_ligs = sorted(ligs, key=lambda x: x[2])
        for pdb_code, gnn_rmsd, glide_rmsd in sorted_ligs:
            print(pdb_code, gnn_rmsd, glide_rmsd)

        for lig in pred_dict:
            for i, elem in enumerate(pred_dict[lig]):
                if elem[0] == '{}_lig0'.format(lig):
                    print(lig, i, elem)


if __name__=="__main__":
    main()