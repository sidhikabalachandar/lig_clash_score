"""
The purpose of this code is to create the cumulative frequency and bar graphs

It can be run on sherlock using
/home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python cum_freq.py all_data /home/users/sidhikab/lig_clash_score/src/visualization/run /home/users/sidhikab/lig_clash_score/models/logs/hybrid_score_feat_clustered /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/combined.csv /home/users/sidhikab/lig_clash_score/reports/figures --split balance_clash
/home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python cum_freq.py group_data /home/users/sidhikab/lig_clash_score/src/visualization/run /home/users/sidhikab/lig_clash_score/models/logs/baseline_clustered /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/combined.csv /home/users/sidhikab/lig_clash_score/reports/figures --split balance_clash --index 0
/home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python cum_freq.py check_data /home/users/sidhikab/lig_clash_score/src/visualization/run /home/users/sidhikab/lig_clash_score/models/logs/hybrid_score_feat_clustered /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/combined.csv /home/users/sidhikab/lig_clash_score/reports/figures --split balance_clash
/home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python cum_freq.py analyze /home/users/sidhikab/lig_clash_score/src/visualization/run /home/users/sidhikab/lig_clash_score/models/logs/hybrid_score_feat_clustered /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/combined.csv /home/users/sidhikab/lig_clash_score/reports/figures --split balance_clash
/home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python cum_freq.py rank_vs_score /home/users/sidhikab/lig_clash_score/src/visualization/run /home/users/sidhikab/lig_clash_score/models/logs/without_protein /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/combined.csv /home/users/sidhikab/lig_clash_score/reports/figures --split balance_clash
/home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python cum_freq.py score_vs_rmsd /home/users/sidhikab/lig_clash_score/src/visualization/run /home/users/sidhikab/lig_clash_score/models/logs/baseline_clustered /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/combined.csv /home/users/sidhikab/lig_clash_score/reports/figures --split balance_clash
"""

import argparse
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import statistics

LABELS = ['GNN with correct pose', 'GNN without correct pose', 'Glide with correct pose', 'Glide without correct pose',
          'Score no vdw with correct pose', 'Score no vdw without correct pose', 'Random with correct pose',
          'Random without correct pose']

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
    return label_df[label_df['target'] == pdb]['target_start_glide_score'].iloc[0]

def get_label(pdb, label_df):
    """
    searches for pdb's rmsd in combined rmsd df
    :param pdb: (string) {target}_lig{id}
    :param label_df: (df) combined rmsd df
    :return: (float) rmsd value
    """
    return label_df[label_df['target'] == pdb]['rmsd'].iloc[0]

def get_score_no_vdw(pdb, label_df):
    """
    searches for pdb's rmsd in combined rmsd df
    :param pdb: (string) {target}_lig{id}
    :param label_df: (df) combined rmsd df
    :return: (float) rmsd value
    """
    return label_df[label_df['target'] == pdb]['target_start_score_no_vdw'].iloc[0]

def get_data(dir):
    data = {}
    for file in os.listdir(dir):
        infile = open(os.path.join(dir, file), 'rb')
        group_dict = pickle.load(infile)
        infile.close()
        for lig in group_dict:
            if lig in data:
                data[lig].extend(group_dict[lig])
            else:
                data[lig] = group_dict[lig]

    return data

def find_num_glide(max_poses, sorted_gnn, target):
    for i in range(max_poses, 0, -1):
        if '{}_lig{}'.format(target, i) in [x[0] for x in sorted_gnn]:
            return i

    return 0

def without_correct(data):
    data_without_correct = {}

    for target in data:
        ls = []
        for val in data[target]:
            if val[0].split('_')[-1][:4] != 'lig0':
                ls.append(val)
        data_without_correct[target] = ls

    return data_without_correct

def get_graph_data(gnn_data, glide_data, score_no_vdw_data, max_poses):
    # get list of lowest rmsd out of top max poses for ranked poses
    # index 0 is gnn ranking with ground truth
    # index 1 is gnn ranking without ground truth
    # index 2 is glide ranking with ground truth,
    # index 3 is glide ranking without ground truth
    # index 4 is score no vdw ranking with ground truth,
    # index 5 is score no vdw ranking without ground truth
    # index 6 is random ranking with ground truth,
    # index 7 is random ranking without ground truth
    ls = []

    for i in range(len(LABELS)):
        ls.append([])

    gnn_data_without_correct = without_correct(gnn_data)
    glide_data_without_correct = without_correct(glide_data)
    score_no_vdw_data_without_correct = without_correct(score_no_vdw_data)

    for target in gnn_data:
        # sort data in reverse ligand order (make sure that we don't choose ground truth in tie breakers)
        rev_gnn_data = sorted(gnn_data[target], key=lambda x: x[0], reverse=True)
        rev_glide_data = sorted(glide_data[target], key=lambda x: x[0], reverse=True)
        rev_score_no_vdw_data = sorted(score_no_vdw_data[target], key=lambda x: x[0], reverse=True)
        rev_gnn_data_without_correct = sorted(gnn_data_without_correct[target], key=lambda x: x[0], reverse=True)
        rev_glide_data_without_correct = sorted(glide_data_without_correct[target], key=lambda x: x[0], reverse=True)
        rev_score_no_vdw_data_without_correct = sorted(score_no_vdw_data_without_correct[target], key=lambda x: x[0],
                                                        reverse=True)

        sorted_gnn = sorted(rev_gnn_data, key=lambda x: x[1])
        sorted_glide = sorted(rev_glide_data, key=lambda x: x[1])
        sorted_score_no_vdw = sorted(rev_score_no_vdw_data, key=lambda x: x[1])
        sorted_gnn_without_correct = sorted(rev_gnn_data_without_correct, key=lambda x: x[1])
        sorted_glide_without_correct = sorted(rev_glide_data_without_correct, key=lambda x: x[1])
        score_no_vdw_without_correct = sorted(rev_score_no_vdw_data_without_correct, key=lambda x: x[1])
        num_glide = find_num_glide(max_poses, sorted_gnn, target)
        num_poses = min(num_glide, max_poses)
        ls[0].append(min(sorted_gnn[:num_poses], key=lambda x: x[2])[2])
        ls[1].append(min(sorted_gnn_without_correct[:num_poses], key=lambda x: x[2])[2])
        ls[2].append(min(sorted_glide[:num_poses], key=lambda x: x[2])[2])
        ls[3].append(min(sorted_glide_without_correct[:num_poses], key=lambda x: x[2])[2])
        ls[4].append(min(sorted_score_no_vdw[:num_poses], key=lambda x: x[2])[2])
        ls[5].append(min(score_no_vdw_without_correct[:num_poses], key=lambda x: x[2])[2])

    return ls

def get_random_data(gnn_data, score_list, max_poses):
    gnn_data_without_correct = without_correct(gnn_data)

    for target in tqdm(gnn_data, desc='iterating through ligand-protein pairs to get random data'):
        # take average over 100 random shuffles
        ls_with_correct = [[] for _ in range(1, max_poses)]
        ls_without_correct = [[] for _ in range(1, max_poses)]
        for _ in range(100):
            # obtain shuffle
            random_data = sorted(gnn_data[target], key=lambda x: x[0])
            random.shuffle(random_data)
            random_data_without_correct = sorted(gnn_data_without_correct[target], key=lambda x: x[0])
            random.shuffle(random_data_without_correct)

            # for the same shuffle obtain best rmsd over top max_poses, where max_poses varies from 1 to 100
            for num_poses in range(1, max_poses):
                ls_with_correct[num_poses - 1].append(min(random_data[:num_poses], key=lambda x: x[2])[2])
                ls_without_correct[num_poses - 1].append(min(random_data_without_correct[:num_poses],
                                                             key=lambda x: x[2])[2])

        # take average for each val of max_pose
        for i in range(len(ls_with_correct)):
            score_list[i][6].append(statistics.mean(ls_with_correct[i]))
            score_list[i][7].append(statistics.mean(ls_without_correct[i]))

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

def bar_graph(freq_list, out_dir, name, mode):
    x_labels = [i for i in range(len(freq_list[0]))]
    fig, ax = plt.subplots()
    for i, ls in enumerate(freq_list):
        plt.plot(x_labels, ls, label=LABELS[i])

    ax.legend(loc='lower right')
    ax.set_xlabel('Pose Cutoff')
    ax.set_ylabel('Percent')
    plt.savefig(os.path.join(out_dir, 'glide_vs_gnn_{}_{}.png'.format(name, mode)))

def run_all_data(run_path, log_dir, label_file, out_dir, split, grouped_files, n):
    for i, group in enumerate(grouped_files):
        cmd = 'sbatch -p rondror -t 1:00:00 -o {} --wrap="' \
              '/home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python cum_freq.py group_data ' \
              '{} {} {} {} --split {} --n {} --index {}"'
        os.system(cmd.format(os.path.join(run_path, 'graph_{}{}.out'.format(log_dir.split('/')[-1], i)), run_path,
                             log_dir, label_file, out_dir, split, n, i))

def run_group_data(grouped_files, grouped_y_pred, label_df, index, gnn_dir, glide_dir, score_no_vdw_dir):
    gnn_data = {}
    glide_data = {}
    score_no_vdw_data = {}
    for i, code in tqdm(enumerate(grouped_files[index]), desc='pdb_codes in testing set'):
        target = code.split('_')[2]
        if target not in gnn_data:
            gnn_data[target] = []
            glide_data[target] = []
            score_no_vdw_data[target] = []
        pdb_code = '{}_{}'.format(target, code.split('_')[3])
        rmsd = get_label(pdb_code, label_df)
        # insert pdb code (index 0), gnn score (index 1), rmsd (index 2)
        gnn_data[target].append((pdb_code, grouped_y_pred[index][i], rmsd))
        # insert pdb code (index 0), glide rank (index 1), rmsd (index 2)
        glide_data[target].append((pdb_code, get_glide_score(pdb_code, label_df), rmsd))
        # insert pdb code (index 0), score no vdw (index 1), rmsd (index 2)
        score_no_vdw_data[target].append((pdb_code, get_score_no_vdw(pdb_code, label_df), rmsd))

    outfile = open(os.path.join(gnn_dir, '{}.pkl'.format(index)), 'wb')
    pickle.dump(gnn_data, outfile)
    outfile = open(os.path.join(glide_dir, '{}.pkl'.format(index)), 'wb')
    pickle.dump(glide_data, outfile)
    outfile = open(os.path.join(score_no_vdw_dir, '{}.pkl'.format(index)), 'wb')
    pickle.dump(score_no_vdw_data, outfile)

def run_check_data(grouped_files, gnn_dir, glide_dir, score_no_vdw_dir):
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

    unfinished = []
    for i in range(len(grouped_files)):
        if not os.path.exists(os.path.join(score_no_vdw_dir, '{}.pkl'.format(i))):
            unfinished.append(i)

    print('Missing from score no vdw:', len(unfinished), '/', len(grouped_files))
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
    parser.add_argument('--cutoff', type=int, default=2, help='rmsd accuracy cutoff between predicted ligand pose and '
                                                              'true ligand pose')
    parser.add_argument('--max_poses', type=int, default=100, help='maximum number of glide poses considered')
    parser.add_argument('--mode', type=str, default='test', help='data used to create graph, either train or test')
    args = parser.parse_args()

    random.seed(0)
    gnn_dir = os.path.join(args.log_dir, 'gnn_{}'.format(args.mode))
    glide_dir = os.path.join(args.log_dir, 'glide_{}'.format(args.mode))
    score_no_vdw_dir = os.path.join(args.log_dir, 'score_no_vdw_{}'.format(args.mode))

    if not os.path.exists(args.run_path):
        os.mkdir(args.run_path)

    if not os.path.exists(gnn_dir):
        os.mkdir(gnn_dir)

    if not os.path.exists(glide_dir):
        os.mkdir(glide_dir)

    if not os.path.exists(score_no_vdw_dir):
        os.mkdir(score_no_vdw_dir)

    infile = open(os.path.join(args.log_dir, '{}_loader_codes_{}.pkl'.format(args.mode, args.split)), 'rb')
    codes = pickle.load(infile)
    infile.close()

    infile = open(os.path.join(args.log_dir, '{}_y_pred_{}.pkl'.format(args.mode, args.split)), 'rb')
    y_pred = pickle.load(infile)
    infile.close()

    label_df = pd.read_csv(args.label_file)
    name = args.log_dir.split('/')[-1]

    if args.task == 'all_data':
        grouped_files = group_files(args.n, codes)
        run_all_data(args.run_path, args.log_dir, args.label_file, args.out_dir, args.split, grouped_files, args.n)

    elif args.task == 'group_data':
        grouped_files = group_files(args.n, codes)
        grouped_y_pred = group_files(args.n, y_pred)
        run_group_data(grouped_files, grouped_y_pred, label_df, args.index, gnn_dir, glide_dir, score_no_vdw_dir)

    elif args.task == 'check_data':
        grouped_files = group_files(args.n, codes)
        run_check_data(grouped_files, gnn_dir, glide_dir, score_no_vdw_dir)

    elif args.task == 'analyze':
        gnn_data = get_data(gnn_dir)
        glide_data = get_data(glide_dir)
        score_no_vdw_data = get_data(score_no_vdw_dir)

        score_list = []
        for i in tqdm(range(1, args.max_poses), desc='iterating from 1 to 100 top poses'):
            score_list.append(get_graph_data(gnn_data, glide_data, score_no_vdw_data, i))

        get_random_data(gnn_data, score_list, args.max_poses)

        freq_list = []
        for i in range(len(score_list[0])):
            accuracy_ls = []
            for ls in score_list:
                accuracy_ls.append(len([j for j in ls[i] if j < args.cutoff]) * 100 / len(ls[i]))
            freq_list.append(accuracy_ls)

        bar_graph(freq_list, args.out_dir, name, args.mode)

    elif args.task == 'rank_vs_score':
        fig, ax = plt.subplots()
        gnn_data = get_data(gnn_dir)
        for target in gnn_data:
            sorted_gnn = sorted(gnn_data[target], key=lambda x: (x[1], x[0]))
            scores = [x[1] for x in sorted_gnn]
            rank = [i for i in range(len(scores))]
            plt.scatter(rank, scores, c='blue', s=0.5)

        ax.set_xlabel('Rank')
        ax.set_ylabel('GNN Score')
        plt.savefig(os.path.join(args.out_dir, 'rank_vs_score_{}.png'.format(name)))

    elif args.task == 'score_vs_rmsd':
        fig, ax = plt.subplots()
        gnn_data = get_data(gnn_dir)
        for target in gnn_data:
            sorted_gnn = sorted(gnn_data[target], key=lambda x: (x[1], x[0]))
            scores = [x[1] for x in sorted_gnn]
            rmsd = [x[2] for x in sorted_gnn]
            plt.scatter(scores, rmsd, c='blue', s=0.5)

        ax.set_xlabel('GNN Score')
        ax.set_ylabel('RMSD')
        plt.savefig(os.path.join(args.out_dir, 'score_vs_rmsd_{}.png'.format(name)))

    elif args.task == 'epoch_graphs':
        output_file = os.path.join(args.log_dir, 'output.txt')
        epoch_dict = {}
        with open(output_file, 'r') as output:
            i = 0
            for line in output:
                if i > 2:
                    if i % 2 == 0:
                        data = line.split(',')
                        for d in data:
                            vals = d.split(':')
                            if vals[0].strip() not in epoch_dict:
                                epoch_dict[vals[0].strip()] = []
                            epoch_dict[vals[0].strip()].append(float(vals[1].strip()))
                i += 1
        fig, ax = plt.subplots()
        for mode in ['Train RMSE', 'Val RMSE']:
            plt.plot([i + 1 for i in range(len(epoch_dict[mode]))], epoch_dict[mode], label=mode)
        ax.legend()
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE Loss')
        plt.savefig(os.path.join(args.out_dir, 'rmse_{}.png'.format(name)))

        fig, ax = plt.subplots()
        for mode in ['Pearson R', 'Spearman R']:
            plt.plot([i + 1 for i in range(len(epoch_dict[mode]))], epoch_dict[mode], label=mode)
        ax.legend()
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Correlation')
        plt.savefig(os.path.join(args.out_dir, 'correlation_{}.png'.format(name)))

if __name__=="__main__":
    main()