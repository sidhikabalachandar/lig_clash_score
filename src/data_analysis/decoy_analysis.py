"""
The purpose of this code is to create the split files
It can be run on sherlock using
$ /home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python decoy_analysis.py gnn_dict_all /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data_analysis/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/pdbbind_refined_set_labels.csv /home/users/sidhikab/lig_clash_score/reports/figures /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /home/users/sidhikab/lig_clash_score/src/data_analysis/gnn_code_dict /home/users/sidhikab/lig_clash_score/src/data_analysis/glide_code_dict
$ /home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python decoy_analysis.py gnn_dict_group /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data_analysis/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/pdbbind_refined_set_labels.csv /home/users/sidhikab/lig_clash_score/reports/figures /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /home/users/sidhikab/lig_clash_score/src/data_analysis/gnn_code_dict/205.pkl /home/users/sidhikab/lig_clash_score/src/data_analysis/glide_code_dict --index 205
$ /home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python decoy_analysis.py gnn_dict_check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data_analysis/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/pdbbind_refined_set_labels.csv /home/users/sidhikab/lig_clash_score/reports/figures /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /home/users/sidhikab/lig_clash_score/src/data_analysis/gnn_code_dict /home/users/sidhikab/lig_clash_score/src/data_analysis/glide_code_dict
$ /home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python decoy_analysis.py glide_dict_all /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data_analysis/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/pdbbind_refined_set_labels.csv /home/users/sidhikab/lig_clash_score/reports/figures /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /home/users/sidhikab/lig_clash_score/src/data_analysis/gnn_code_dict /home/users/sidhikab/lig_clash_score/src/data_analysis/glide_code_dict
$ /home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python decoy_analysis.py glide_dict_group /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data_analysis/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/pdbbind_refined_set_labels.csv /home/users/sidhikab/lig_clash_score/reports/figures /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /home/users/sidhikab/lig_clash_score/src/data_analysis/gnn_code_dict /home/users/sidhikab/lig_clash_score/src/data_analysis/glide_code_dict/0.pkl --index 0
$ /home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python decoy_analysis.py glide_dict_check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data_analysis/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/pdbbind_refined_set_labels.csv /home/users/sidhikab/lig_clash_score/reports/figures /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /home/users/sidhikab/lig_clash_score/src/data_analysis/gnn_code_dict /home/users/sidhikab/lig_clash_score/src/data_analysis/glide_code_dict
$ /home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python decoy_analysis.py graph /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data_analysis/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/pdbbind_refined_set_labels.csv /home/users/sidhikab/lig_clash_score/reports/figures /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /home/users/sidhikab/lig_clash_score/src/data_analysis/gnn_code_dict /home/users/sidhikab/lig_clash_score/src/data_analysis/glide_code_dict

"""

import argparse
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import seaborn as sns
import random
import sys

sys.path[-2] = '/home/users/sidhikab/lig_clash_score/src'

from atom3d.protein_ligand.get_labels import get_label

MAX_POSES = 100
CUTOFF = 2
LABELS =['gnn with correct pose', 'gnn without correct pose', 'glide']
N = 3

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

def bar_graph(all_freq_ls, save_root):
    sns.set_context("talk", font_scale=0.8)
    unnormalized_all_graph_rmsds = []

    label = 'Best over \nall sampled \nposes'

    unnormalized_all_graph_rmsds.append([label, all_freq_ls[0], 'GNN with correct pose'])
    unnormalized_all_graph_rmsds.append([label, all_freq_ls[1], 'GNN without correct pose'])
    unnormalized_all_graph_rmsds.append([label, all_freq_ls[2], 'Glide'])

    df = pd.DataFrame(unnormalized_all_graph_rmsds)
    df.columns = ['Type', 'Percent', 'Legend']
    g = sns.catplot(x='Type', y='Percent', hue='Legend', data=df, kind="bar")
    # plt.title('Unormalized')
    ax = plt.gca()
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.savefig(os.path.join(save_root, 'glide_vs_gnn.png'))

def get_prots(docked_prot_file):
    """
    gets list of all protein, target ligands, and starting ligands in the index file
    :param docked_prot_file: (string) file listing proteins to process
    :return: process (list) list of all protein, target ligands, and starting ligands to process
    """
    process = []
    with open(docked_prot_file) as fp:
        for line in fp:
            if line[0] == '#': continue
            protein, target, start = line.strip().split()
            process.append((protein, target, start))

    return process


def group_files(n, process):
    """
    groups pairs into sublists of size n
    :param n: (int) sublist size
    :param process: (list) list of pairs to process
    :return: grouped_files (list) list of sublists of pairs
    """
    grouped_files = []

    for i in range(0, len(process), n):
        grouped_files += [process[i: i + n]]

    return grouped_files

def get_gnn_code_dict(process, pkl_file, label_file, raw_root):
    """
    gets list of all protein, target ligands, starting ligands, and starting indices information in the index file (up to
    CUTOFF)
    :param process: (list) shuffled list of all protein, target ligands, and starting ligands to process
    :param pkl_file: (string) file containing list of all protein, target ligands, starting ligands, and starting
                                indices information (or file path where this information will be saved)
    :param label_file: (string) file containing rmsd label information
    :param raw_root: (string) path to directory with data
    :return: grouped_files (list) list of all protein, target ligands, starting ligands, and starting indices to process
    """
    label_df = pd.read_csv(label_file)
    gnn_code_dict = {}

    for protein, target, start in tqdm(process, desc='going through protein, target, start groups'):
        if (protein, target, start) not in gnn_code_dict:
            gnn_code_dict[(protein, target, start)] = []

        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
        graph_dir = '{}/{}-to-{}_graph.pkl'.format(pair_path, target, start)
        infile = open(graph_dir, 'rb')
        graph_data = pickle.load(infile)
        infile.close()
        for pdb_code in graph_data:
            if len(label_df[label_df['target'] == pdb_code]) != 0:
                gnn_code_dict[(protein, target, start)].append((pdb_code, get_label(pdb_code, label_df)))

    outfile = open(pkl_file, 'wb')
    pickle.dump(gnn_code_dict, outfile)

    return gnn_code_dict

def combine_code_dict(gnn_code_dict_root):
    """
    gets list of all protein, target ligands, starting ligands, and starting indices information in the index file (up to
    CUTOFF)
    :param process: (list) shuffled list of all protein, target ligands, and starting ligands to process
    :param pkl_file: (string) file containing list of all protein, target ligands, starting ligands, and starting
                                indices information (or file path where this information will be saved)
    :param label_file: (string) file containing rmsd label information
    :param raw_root: (string) path to directory with data
    :return: grouped_files (list) list of all protein, target ligands, starting ligands, and starting indices to process
    """
    gnn_code_dict = {}
    for file in os.listdir(gnn_code_dict_root):
        infile = open(os.path.join(gnn_code_dict_root, file), 'rb')
        in_dict = pickle.load(infile)
        infile.close()
        gnn_code_dict.update(in_dict)

    return gnn_code_dict

def get_glide_code_dict(process, pkl_file, label_file, raw_root):
    """
    gets list of all protein, target ligands, starting ligands, and starting indices information in the index file (up to
    CUTOFF)
    :param process: (list) shuffled list of all protein, target ligands, and starting ligands to process
    :param pkl_file: (string) file containing list of all protein, target ligands, starting ligands, and starting
                                indices information (or file path where this information will be saved)
    :param label_file: (string) file containing rmsd label information
    :param raw_root: (string) path to directory with data
    :return: grouped_files (list) list of all protein, target ligands, starting ligands, and starting indices to process
    """
    label_df = pd.read_csv(label_file)
    glide_code_dict = {}

    for protein, target, start in tqdm(process, desc='going through protein, target, start groups'):
        if (protein, target, start) not in glide_code_dict:
            glide_code_dict[(protein, target, start)] = []

        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
        pose_path = os.path.join(pair_path, 'ligand_poses')
        for i in range(1, MAX_POSES):
            pdb_code = '{}_lig{}'.format(target, i)
            if os.path.exists(os.path.join(pose_path, '{}.sdf'.format(pdb_code))) and \
                    len(label_df[label_df['target'] == pdb_code]) != 0:
                glide_code_dict[(protein, target, start)].append((pdb_code, get_label(pdb_code, label_df)))

    outfile = open(pkl_file, 'wb')
    pickle.dump(glide_code_dict, outfile)

    return glide_code_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='file listing proteins to process')
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('run_path', type=str, help='file listing proteins to process')
    parser.add_argument('label_file', type=str, help='file listing proteins to process')
    parser.add_argument('graph_save_root', type=str, help='file listing proteins to process')
    parser.add_argument('raw_root', type=str, help='file listing proteins to process')
    parser.add_argument('gnn_code_dict', type=str)
    parser.add_argument('glide_code_dict', type=str)
    parser.add_argument('--index', type=int, default=-1)
    args = parser.parse_args()

    random.seed(0)
    if not os.path.exists(args.run_path):
        print(args.run_path)
        os.mkdir(args.run_path)

    if args.task == 'gnn_dict_all':
        process = get_prots(args.docked_prot_file)
        random.shuffle(process)
        grouped_files = group_files(N, process)

        if not os.path.exists(args.run_path):
            os.mkdir(args.run_path)

        for i, group in enumerate(grouped_files):
            cmd = 'sbatch -p owners -t 1:00:00 -o {} --wrap="' \
                  '/home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python decoy_analysis.py ' \
                  'gnn_dict_group {} {} {} {} {} {}/{}.pkl {} --index {}"'
            os.system(cmd.format(os.path.join(args.run_path, 'gnn_dict{}.out'.format(i)), args.docked_prot_file,
                                 args.run_path, args.label_file, args.graph_save_root, args.raw_root,
                                 args.gnn_code_dict, i, args.glide_code_dict, i))
            # print(cmd.format(os.path.join(args.run_path, 'gnn_dict{}.out'.format(i)), args.docked_prot_file,
            #                      args.run_path, args.label_file, args.graph_save_root, args.raw_root,
            #                      args.gnn_code_dict, i, args.glide_code_dict, i))
            
    if args.task == 'gnn_dict_group':
        process = get_prots(args.docked_prot_file)
        random.shuffle(process)
        grouped_files = group_files(N, process)
        get_gnn_code_dict(grouped_files[args.index], args.gnn_code_dict, args.label_file, args.raw_root)

    if args.task == 'gnn_dict_check':
        process = get_prots(args.docked_prot_file)
        random.shuffle(process)
        grouped_files = group_files(N, process)
        unfinished = []
        for i in range(len(grouped_files)):
            if not os.path.exists('{}/{}.pkl'.format(args.gnn_code_dict, i)):
                unfinished.append(i)

        print('Missing', len(unfinished), '/', len(grouped_files))
        print(unfinished)

    if args.task == 'glide_dict_all':
        process = get_prots(args.docked_prot_file)
        random.shuffle(process)
        grouped_files = group_files(N, process)

        if not os.path.exists(args.run_path):
            os.mkdir(args.run_path)

        for i, group in enumerate(grouped_files):
            cmd = 'sbatch -p owners -t 1:00:00 -o {} --wrap="' \
                  '/home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python decoy_analysis.py ' \
                  'glide_dict_group {} {} {} {} {} {} {}/{}.pkl --index {}"'
            os.system(cmd.format(os.path.join(args.run_path, 'gnn_dict{}.out'.format(i)), args.docked_prot_file,
                                 args.run_path, args.label_file, args.graph_save_root, args.raw_root,
                                 args.gnn_code_dict, args.glide_code_dict, i, i))
    if args.task == 'glide_dict_group':
        process = get_prots(args.docked_prot_file)
        random.shuffle(process)
        grouped_files = group_files(N, process)
        get_glide_code_dict(grouped_files[args.index], args.glide_code_dict, args.label_file, args.raw_root)

    if args.task == 'glide_dict_check':
        process = get_prots(args.docked_prot_file)
        random.shuffle(process)
        grouped_files = group_files(N, process)
        unfinished = []
        for i in range(len(grouped_files)):
            if not os.path.exists('{}/{}.pkl'.format(args.glide_code_dict, i)):
                unfinished.append(i)

        print('Missing', len(unfinished), '/', len(grouped_files))
        print(unfinished)

    if args.task == 'graph':
        process = get_prots(args.docked_prot_file)
        random.shuffle(process)
        gnn_code_dict = combine_code_dict(args.gnn_code_dict)
        glide_code_dict = combine_code_dict(args.glide_code_dict)

        # index 0 is gnn pred index, 1 is gnn without ground truth, 2 is glide
        all_ls = [[], [], []]
        error_count = 0
        for protein, target, start in gnn_code_dict:
            if len(glide_code_dict[(protein, target, start)]) != 0:
                all_ls[0].append(min(gnn_code_dict[(protein, target, start)], key=lambda x: x[1])[1])

                if min(gnn_code_dict[(protein, target, start)], key=lambda x: x[1])[1] == 0:
                    all_ls[1].append(sorted(gnn_code_dict[(protein, target, start)], key=lambda x: x[1])[1][1])
                else:
                    all_ls[1].append(min(gnn_code_dict[(protein, target, start)], key=lambda x: x[1])[1])

                all_ls[2].append(min(glide_code_dict[(protein, target, start)], key=lambda x: x[1])[1])
            else:
                error_count += 1

        print('Error count =', error_count)

        all_freq_ls = []
        for i in range(len(all_ls)):
            all_freq_ls.append(len([j for j in all_ls[i] if j < CUTOFF]) * 100 / len(all_ls[i]))

        print("Labels:", LABELS)
        print("All frequencies:", all_freq_ls)

        graph('all', all_ls, args.graph_save_root)
        bar_graph(all_freq_ls, args.graph_save_root)

if __name__=="__main__":
    main()