"""
The purpose of this code is to create the cumulative frequency and bar graphs

It can be run on sherlock using
/home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python cum_freq.py all_data /home/users/sidhikab/lig_clash_score/src/visualization/run /home/users/sidhikab/lig_clash_score/models/logs/baseline_clustered /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/combined.csv /home/users/sidhikab/lig_clash_score/reports/figures --gnn_dir /home/users/sidhikab/lig_clash_score/models/logs/baseline_clustered/gnn --glide_dir /home/users/sidhikab/lig_clash_score/models/logs/baseline_clustered/glide --split balance_clash
/home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python cum_freq.py group_data /home/users/sidhikab/lig_clash_score/src/visualization/run /home/users/sidhikab/lig_clash_score/models/logs/baseline_clustered /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/combined.csv /home/users/sidhikab/lig_clash_score/reports/figures --gnn_dir /home/users/sidhikab/lig_clash_score/models/logs/baseline_clustered/gnn --glide_dir /home/users/sidhikab/lig_clash_score/models/logs/baseline_clustered/glide --split balance_clash --index 0
/home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python cum_freq.py check_data /home/users/sidhikab/lig_clash_score/src/visualization/run /home/users/sidhikab/lig_clash_score/models/logs/baseline_clustered /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/combined.csv /home/users/sidhikab/lig_clash_score/reports/figures --gnn_dir /home/users/sidhikab/lig_clash_score/models/logs/baseline_clustered/gnn --glide_dir /home/users/sidhikab/lig_clash_score/models/logs/baseline_clustered/glide --split balance_clash
/home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python cum_freq.py analyze /home/users/sidhikab/lig_clash_score/src/visualization/run /home/users/sidhikab/lig_clash_score/models/logs/baseline_clustered /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/combined.csv /home/users/sidhikab/lig_clash_score/reports/figures --gnn_dir /home/users/sidhikab/lig_clash_score/models/logs/baseline_clustered/gnn --glide_dir /home/users/sidhikab/lig_clash_score/models/logs/baseline_clustered/glide --split balance_clash
"""

import argparse
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

LABELS = ['GNN with correct pose', 'GNN without correct pose', 'Glide with correct pose', 'Glide without correct pose']

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

def get_glide_score(pdb, label_df):
    """
    searches for pdb's rmsd in combined rmsd df
    :param pdb: (string) {target}_lig{id}
    :param label_df: (df) combined rmsd df
    :return: (float) rmsd value
    """
    return label_df[label_df['target'] == pdb]['glide_score'].iloc[0]

def get_label(pdb, label_df):
    """
    searches for pdb's rmsd in combined rmsd df
    :param pdb: (string) {target}_lig{id}
    :param label_df: (df) combined rmsd df
    :return: (float) rmsd value
    """
    return label_df[label_df['target'] == pdb]['rmsd'].iloc[0]

def get_glide_data(glide_dir):
    glide_data = {}
    for file in os.listdir(glide_dir):
        infile = open(os.path.join(glide_dir, file), 'rb')
        group_glide_dict = pickle.load(infile)
        infile.close()
        for lig in group_glide_dict:
            if lig in glide_data:
                glide_data[lig].extend(group_glide_dict[lig])
            else:
                glide_data[lig] = group_glide_dict[lig]

    return glide_data

def get_gnn_data(gnn_dir):
    gnn_data = {}
    for file in os.listdir(gnn_dir):
        infile = open(os.path.join(gnn_dir, file), 'rb')
        group_gnn_dict = pickle.load(infile)
        infile.close()
        for lig in group_gnn_dict:
            if lig in gnn_data:
                gnn_data[lig].extend(group_gnn_dict[lig])
            else:
                gnn_data[lig] = group_gnn_dict[lig]

    return gnn_data

def get_stats(sorted_data):
    i = 0
    with_ground_truth = sorted_data[i][2]
    code = sorted_data[i][0]
    while code.split('_')[-1][:4] == 'lig0':
        i += 1
        code = sorted_data[i][0]
    without_ground_truth = sorted_data[i][2]
    return with_ground_truth, without_ground_truth, sorted_data[i]

def get_graph_data(gnn_data, glide_data):
    # index 0 is gnn with ground truth, index 1 is gnn without ground truth, index 2 is glide with ground truth,
    # index 3 is glide without ground truth
    top_1_ls = []
    top_100_ls = []
    all_ls = []

    for i in range(len(LABELS)):
        top_1_ls.append([])
        top_100_ls.append([])
        all_ls.append([])

    print(len(gnn_data))
    vals = []
    for target in tqdm(gnn_data, desc='target ligands in gnn_data'):
        target = '5gsa'
        sorted_gnn = sorted(gnn_data[target], key=lambda x: x[1])
        with_ground_truth, without_ground_truth, val = get_stats(sorted_gnn)
        top_1_ls[0].append(with_ground_truth)
        top_1_ls[1].append(without_ground_truth)
        sorted_glide = sorted(glide_data[target], key=lambda x: x[1])
        with_ground_truth, without_ground_truth, val = get_stats(sorted_glide)
        top_1_ls[2].append(with_ground_truth)
        top_1_ls[3].append(without_ground_truth)

        sorted_100_gnn = sorted(sorted_gnn[:100], key=lambda x: x[2])
        with_ground_truth, without_ground_truth, val = get_stats(sorted_100_gnn)
        top_100_ls[0].append(with_ground_truth)
        top_100_ls[1].append(without_ground_truth)
        sorted_100_glide = sorted(sorted_glide[:100], key=lambda x: x[2])
        print(sorted_100_glide)
        with_ground_truth, without_ground_truth, val = get_stats(sorted_100_glide)
        top_100_ls[2].append(with_ground_truth)
        top_100_ls[3].append(without_ground_truth)
        vals.append(val)

        sorted_all_gnn = sorted(gnn_data[target], key=lambda x: x[2])
        with_ground_truth, without_ground_truth, val = get_stats(sorted_all_gnn)
        all_ls[0].append(with_ground_truth)
        all_ls[1].append(without_ground_truth)
        sorted_all_glide = sorted(glide_data[target], key=lambda x: x[2])
        with_ground_truth, without_ground_truth, val = get_stats(sorted_all_glide)
        all_ls[2].append(with_ground_truth)
        all_ls[3].append(without_ground_truth)
        break

    outfile = open('hybrid_val.pkl', 'wb')
    pickle.dump(vals, outfile)
    return top_1_ls, top_100_ls, all_ls

def graph(title, ls, out_dir, name):
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

    plt.savefig(os.path.join(out_dir, '{}_{}.png'.format(title, name)))

def bar_graph(top_1_freq_ls, top_100_freq_ls, all_freq_ls, out_dir, name):
    sns.set_context("talk", font_scale=0.8)
    unnormalized_all_graph_rmsds = []

    label1 = 'Top \ncorrect'
    label2 = 'Best over \ntop 100'
    label3 = 'Best over \nall sampled \nposes'

    for i in range(len(LABELS)):
        unnormalized_all_graph_rmsds.append([label1, top_1_freq_ls[i], LABELS[i]])
        unnormalized_all_graph_rmsds.append([label2, top_100_freq_ls[i], LABELS[i]])
        unnormalized_all_graph_rmsds.append([label3, all_freq_ls[i], LABELS[i]])

    df = pd.DataFrame(unnormalized_all_graph_rmsds)
    df.columns = ['Type', 'Percent', 'Legend']
    sns.catplot(x='Type', y='Percent', hue='Legend', data=df, kind="bar")
    ax = plt.gca()
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.savefig(os.path.join(out_dir, 'glide_vs_gnn_{}.png'.format(name)))

def run_all_data(run_path, log_dir, label_file, out_dir, split, gnn_dir, glide_dir, grouped_files, n):
    for i, group in enumerate(grouped_files):
        cmd = 'sbatch -p owners -t 1:00:00 -o {} --wrap="' \
              '/home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python cum_freq.py group_data ' \
              '{} {} {} {} --split {} --n {}  --gnn_dir {} --glide_dir {} --index {}"'
        os.system(cmd.format(os.path.join(run_path, 'graph_{}{}.out'.format(log_dir.split('/')[-1], i)), run_path,
                             log_dir, label_file, out_dir, split, n, gnn_dir, glide_dir, i))

def run_group_data(grouped_files, grouped_y_pred, label_df, index, gnn_dir, glide_dir):
    gnn_data = {}
    glide_data = {}
    for i, code in tqdm(enumerate(grouped_files[index]), desc='pdb_codes in testing set'):
        target = code.split('_')[2]
        if target not in gnn_data:
            gnn_data[target] = []
            glide_data[target] = []
        pdb_code = '{}_{}'.format(target, code.split('_')[3])
        # insert pdb code (index 0), gnn score (index 1), rmsd (index 2)
        gnn_data[target].append((pdb_code, grouped_y_pred[index][i], get_label(pdb_code, label_df)))
        # insert pdb code (index 0), glide rank (index 1), rmsd (index 2)
        glide_data[target].append((pdb_code, get_glide_score(pdb_code, label_df), get_label(pdb_code, label_df)))

    outfile = open(os.path.join(gnn_dir, '{}.pkl'.format(index)), 'wb')
    pickle.dump(gnn_data, outfile)
    outfile = open(os.path.join(glide_dir, '{}.pkl'.format(index)), 'wb')
    pickle.dump(glide_data, outfile)

def run_check_data(grouped_files, gnn_dir, glide_dir):
    unfinished = []
    for i in range(len(grouped_files)):
        if not os.path.exists(os.path.join(gnn_dir, '{}.pkl'.format(i))):
            unfinished.append(i)

    print('Missing from gnn:', len(unfinished), '/', len(grouped_files))
    print(unfinished)

    unfinished = []
    for i in range(len(grouped_files)):
        if not os.path.exists(os.path.join(glide_dir, '{}.pkl'.format(i))):
            unfinished.append(i)

    print('Missing from glide:', len(unfinished), '/', len(grouped_files))
    print(unfinished)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either all, group, check, '
                                               'all_dist_check, group_dist_check, check_dist_check, '
                                               'all_name_check, group_name_check, check_name_check, or delete')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('log_dir', type=str, default=None, help='specific logging directory')
    parser.add_argument('label_file', type=str, help='file with rmsd labels')
    parser.add_argument('out_dir', type=str, help='directory where all graphs will be saved')
    parser.add_argument('--split', type=str, default='random', help='split name')
    parser.add_argument('--index', type=int, default=-1, help='for group task, group number')
    parser.add_argument('--n', type=int, default=100, help='number of protein, target, start groups processed in '
                                                         'group task')
    parser.add_argument('--gnn_dir', type=str, default=os.path.join(os.getcwd(), 'gnn'), help='directory in which gnn '
                                                                                              'data saved')
    parser.add_argument('--glide_dir', type=str, default=os.path.join(os.getcwd(), 'glide'), help='directory in which gnn '
                                                                                              'data saved')
    parser.add_argument('--cutoff', type=int, default=2, help='rmsd accuracy cutoff between predicted ligand pose and '
                                                              'true ligand pose')
    args = parser.parse_args()

    if not os.path.exists(args.run_path):
        os.mkdir(args.run_path)

    if not os.path.exists(args.gnn_dir):
        os.mkdir(args.gnn_dir)

    if not os.path.exists(args.glide_dir):
        os.mkdir(args.glide_dir)

    infile = open(os.path.join(args.log_dir, 'test_loader_codes_{}.pkl'.format(args.split)), 'rb')
    codes = pickle.load(infile)
    infile.close()

    infile = open(os.path.join(args.log_dir, 'y_pred_{}.pkl'.format(args.split)), 'rb')
    y_pred = pickle.load(infile)
    infile.close()

    label_df = pd.read_csv(args.label_file)
    name = args.log_dir.split('/')[-1]

    if args.task == 'all_data':
        grouped_files = group_files(args.n, codes)
        run_all_data(args.run_path, args.log_dir, args.label_file, args.out_dir, args.split, args.gnn_dir,
                     args.glide_dir, grouped_files, args.n)

    elif args.task == 'group_data':
        grouped_files = group_files(args.n, codes)
        grouped_y_pred = group_files(args.n, y_pred)
        run_group_data(grouped_files, grouped_y_pred, label_df, args.index, args.gnn_dir, args.glide_dir)

    elif args.task == 'check_data':
        grouped_files = group_files(args.n, codes)
        run_check_data(grouped_files, args.gnn_dir, args.glide_dir)

    elif args.task == 'analyze':
        gnn_data = get_gnn_data(args.gnn_dir)
        glide_data = get_glide_data(args.glide_dir)

        top_1_ls, top_100_ls, all_ls = get_graph_data(gnn_data, glide_data)

        top_1_freq_ls = []
        top_100_freq_ls = []
        all_freq_ls = []
        for i in range(len(top_1_ls)):
            top_1_freq_ls.append(len([j for j in top_1_ls[i] if j < args.cutoff]) * 100 / len(top_1_ls[i]))
            top_100_freq_ls.append(len([j for j in top_100_ls[i] if j < args.cutoff]) * 100 / len(top_100_ls[i]))
            all_freq_ls.append(len([j for j in all_ls[i] if j < args.cutoff]) * 100 / len(all_ls[i]))

        print("Labels:", LABELS)
        print("Top 1 frequencies:", top_1_freq_ls)
        print("Top 100 frequencies:", top_100_freq_ls)
        print("All frequency:", all_freq_ls)

        graph('top_1', top_1_ls, args.out_dir, name)
        graph('top_100', top_100_ls, args.out_dir, name)
        graph('all', all_ls, args.out_dir, name)

        bar_graph(top_1_freq_ls, top_100_freq_ls, all_freq_ls, args.out_dir, name)

if __name__=="__main__":
    main()