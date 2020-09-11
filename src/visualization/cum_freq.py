"""
The purpose of this code is to create the cumulative frequency and bar graphs

It can be run on sherlock using
/home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python cum_freq.py /home/users/sidhikab/lig_clash_score/models/logs/v1 /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/pdbbind_refined_set_labels.csv /home/users/sidhikab/lig_clash_score/reports/figures
"""

import argparse
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import sys
sys.path[-2] = '/home/users/sidhikab/lig_clash_score/src'
from data.get_labels import get_label

MAX_POSES= 100
CUTOFF = 2
LABELS = ['gnn with ground truth', 'gnn without ground truth', 'glide']

def get_glide_data(gnn_data, label_df, log_dir):
    save_file = os.path.join(log_dir, 'glide_data.pkl')
    if os.path.exists(save_file):
        infile = open(save_file, 'rb')
        glide_data = pickle.load(infile)
        infile.close()
    else:
        glide_data = {}
        for target in tqdm(gnn_data, desc='target ligands in gnn_data'):
            if target not in glide_data:
                glide_data[target] = []

            for i in range(1, MAX_POSES):
                pdb_code = '{}_lig{}'.format(target, i)
                if len(label_df[label_df['target'] == pdb_code]) != 0:
                    # insert pdb code (index 0), glide rank (index 1), rmsd (index 2)
                    glide_data[target].append((pdb_code, i, get_label(pdb_code, label_df)))

        outfile = open(save_file, 'wb')
        pickle.dump(glide_data, outfile)

    return glide_data

def get_gnn_data(codes, y_pred, label_df, log_dir):
    save_file = os.path.join(log_dir, 'gnn_data.pkl')
    if os.path.exists(save_file):
        infile = open(save_file, 'rb')
        gnn_data = pickle.load(infile)
        infile.close()
    else:
        gnn_data = {}
        for i, code in tqdm(enumerate(codes), desc='pdb_codes in testing set'):
            target = code.split('_')[2]
            if target not in gnn_data:
                gnn_data[target] = []
            pdb_code = '{}_{}'.format(target, code.split('_')[3])
            # insert pdb code (index 0), gnn score (index 1), rmsd (index 2)
            gnn_data[target].append((pdb_code, y_pred[i], get_label(pdb_code, label_df)))

        outfile = open(save_file, 'wb')
        pickle.dump(gnn_data, outfile)

    return gnn_data

def get_graph_data(gnn_data, glide_data):
    # index 0 is gnn with ground truth, index 1 is gnn without ground truth, index 2 is glide
    top_1_ls = [[], [], []]
    top_100_ls = [[], [], []]
    all_ls = [[], [], []]

    for target in tqdm(gnn_data, desc='target ligands in gnn_data'):
        sorted_gnn = sorted(gnn_data[target], key=lambda x: x[1])
        top_1_ls[0].append(sorted_gnn[0][2])
        if sorted_gnn[0][2] == 0:
            top_1_ls[1].append(sorted_gnn[1][2])
        else:
            top_1_ls[1].append(sorted_gnn[0][2])
        top_1_ls[2].append(min(glide_data[target], key=lambda x: x[1])[2])

        min_100_gnn = min(sorted_gnn[:len(glide_data[target])], key=lambda x: x[2])[2]
        top_100_ls[0].append(min_100_gnn)
        if min_100_gnn == 0:
            top_100_ls[1].append(sorted(sorted_gnn[:len(glide_data[target])], key=lambda x: x[2])[1][2])
        else:
            top_100_ls[1].append(min_100_gnn)
        top_100_ls[2].append(min(glide_data[target], key=lambda x: x[2])[2])

        min_all_gnn = min(gnn_data[target], key=lambda x: x[2])[2]
        all_ls[0].append(min_all_gnn)
        if min_all_gnn == 0:
            all_ls[1].append(sorted(gnn_data[target], key=lambda x: x[2])[1][2])
        else:
            all_ls[1].append(min_all_gnn)
        all_ls[2].append(min(glide_data[target], key=lambda x: x[2])[2])

    return top_1_ls, top_100_ls, all_ls

def graph(title, ls, out_dir):
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

    plt.savefig(os.path.join(out_dir, '{}.png'.format(title)))

def bar_graph(top_1_freq_ls, top_100_freq_ls, all_freq_ls, out_dir):
    sns.set_context("talk", font_scale=0.8)
    unnormalized_all_graph_rmsds = []

    label1 = 'Top \ncorrect'
    label2 = 'Best over \ntop 100'
    label3 = 'Best over \nall sampled \nposes'

    unnormalized_all_graph_rmsds.append([label1, top_1_freq_ls[0], 'GNN with correct pose'])
    unnormalized_all_graph_rmsds.append([label1, top_1_freq_ls[1], 'GNN without correct pose'])
    unnormalized_all_graph_rmsds.append([label1, top_1_freq_ls[2], 'Glide'])

    unnormalized_all_graph_rmsds.append([label2, top_100_freq_ls[0], 'GNN with correct pose'])
    unnormalized_all_graph_rmsds.append([label2, top_100_freq_ls[1], 'GNN without correct pose'])
    unnormalized_all_graph_rmsds.append([label2, top_100_freq_ls[2], 'Glide'])

    unnormalized_all_graph_rmsds.append([label3, all_freq_ls[0], 'GNN with correct pose'])
    unnormalized_all_graph_rmsds.append([label3, all_freq_ls[1], 'GNN without correct pose'])
    unnormalized_all_graph_rmsds.append([label3, all_freq_ls[2], 'Glide'])

    df = pd.DataFrame(unnormalized_all_graph_rmsds)
    df.columns = ['Type', 'Percent', 'Legend']
    sns.catplot(x='Type', y='Percent', hue='Legend', data=df, kind="bar")
    # plt.title('Unormalized')
    ax = plt.gca()
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.savefig(os.path.join(out_dir, 'glide_vs_gnn.png'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_dir', type=str, default=None, help='specific logging directory')
    parser.add_argument('label_file', type=str, help='file with rmsd labels')
    parser.add_argument('out_dir', type=str, help='directory where all graphs will be saved')
    parser.add_argument('--split', type=str, default='random', help='split name')
    args = parser.parse_args()

    infile = open(os.path.join(args.log_dir, 'test_loader_codes_{}.pkl'.format(args.split)), 'rb')
    codes = pickle.load(infile)
    infile.close()

    infile = open(os.path.join(args.log_dir, 'y_pred_{}.pkl'.format(args.split)), 'rb')
    y_pred = pickle.load(infile)
    infile.close()

    label_df = pd.read_csv(args.label_file)

    gnn_data = get_gnn_data(codes, y_pred, label_df, args.log_dir)
    glide_data = get_glide_data(gnn_data, label_df, args.log_dir)

    top_1_ls, top_100_ls, all_ls = get_graph_data(gnn_data, glide_data)

    top_1_freq_ls = []
    top_100_freq_ls = []
    all_freq_ls = []
    for i in range(len(top_1_ls)):
        top_1_freq_ls.append(len([j for j in top_1_ls[i] if j < CUTOFF]) * 100 / len(top_1_ls[i]))
        top_100_freq_ls.append(len([j for j in top_100_ls[i] if j < CUTOFF]) * 100 / len(top_100_ls[i]))
        all_freq_ls.append(len([j for j in all_ls[i] if j < CUTOFF]) * 100 / len(all_ls[i]))

    print("Labels:", LABELS)
    print("Top 1 frequencies:", top_1_freq_ls)
    print("Top 100 frequencies:", top_100_freq_ls)
    print("All frequency:", all_freq_ls)

    graph('top_1', top_1_ls, args.out_dir)
    graph('top_100', top_100_ls, args.out_dir)
    graph('all', all_ls, args.out_dir)

    bar_graph(top_1_freq_ls, top_100_freq_ls, all_freq_ls, args.out_dir)

if __name__=="__main__":
    main()