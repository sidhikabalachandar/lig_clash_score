"""
The purpose of this code is to create the cumulative frequency and bar graphs

It can be run on sherlock using
/home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python score_no_vdw_analysis.py all_data /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random_clash.txt /home/users/sidhikab/lig_clash_score/src/visualization/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/combined.csv /home/users/sidhikab/lig_clash_score/src/visualization/score_no_vdw /home/users/sidhikab/lig_clash_score/src/visualization/glide /home/users/sidhikab/lig_clash_score/reports/figures
/home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python score_no_vdw_analysis.py all_data /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random_clash.txt /home/users/sidhikab/lig_clash_score/src/visualization/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/combined.csv /home/users/sidhikab/lig_clash_score/src/visualization/score_no_vdw /home/users/sidhikab/lig_clash_score/src/visualization/glide /home/users/sidhikab/lig_clash_score/reports/figures --index 0
/home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python score_no_vdw_analysis.py check_data /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random_clash.txt /home/users/sidhikab/lig_clash_score/src/visualization/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/combined.csv /home/users/sidhikab/lig_clash_score/src/visualization/score_no_vdw /home/users/sidhikab/lig_clash_score/src/visualization/glide /home/users/sidhikab/lig_clash_score/reports/figures
/home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python score_no_vdw_analysis.py analyze /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random_clash.txt /home/users/sidhikab/lig_clash_score/src/visualization/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/combined.csv /home/users/sidhikab/lig_clash_score/src/visualization/score_no_vdw /home/users/sidhikab/lig_clash_score/src/visualization/glide /home/users/sidhikab/lig_clash_score/reports/figures
"""

import argparse
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

LABELS = ['Score no vdw with correct pose', 'Score no vdw without correct pose', 'Glide with correct pose',
          'Glide without correct pose', 'Score no vdw with correct pose only clash',
          'Score no vdw without correct pose only clash', 'Glide with correct pose only clash',
          'Glide without correct pose only clash']

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
    return label_df[label_df['target'] == pdb]['score_no_vdw'].iloc[0]

def get_glide_score(pdb, label_df):
    """
    searches for pdb's rmsd in combined rmsd df
    :param pdb: (string) {target}_lig{id}
    :param label_df: (df) combined rmsd df
    :return: (float) rmsd value
    """
    return label_df[label_df['target'] == pdb]['glide_score'].iloc[0]

def get_data(dir):
    data = {}
    for file in os.listdir(dir):
        infile = open(os.path.join(dir, file), 'rb')
        group_dict = pickle.load(infile)
        infile.close()
        for pair in group_dict:
            if len(group_dict[pair]) > 0:
                if pair in data:
                    data[pair].extend(group_dict[pair])
                else:
                    data[pair] = group_dict[pair]

    return data

def find_num_glide(max_poses, sorted_gnn, target):
    for i in range(max_poses, 0, -1):
        if '{}_lig{}'.format(target, i) in [x[0] for x in sorted_gnn]:
            return i

    return 0

def without_correct(data):
    data_without_correct = {}

    for pair in data:
        ls = []
        for val in data[pair]:
            if val[0].split('_')[-1][:4] != 'lig0':
                ls.append(val)
        data_without_correct[pair] = ls

    return data_without_correct

def get_graph_data(score_no_vdw_data, glide_data, process, max_poses):
    # get list of lowest rmsd out of top max poses for ranked poses
    # index 0 is score no vdw ranking with ground truth,
    # index 1 is score no vdw ranking without ground truth,
    # index 2 is glide ranking with ground truth,
    # index 3 is glide ranking without ground truth
    # index 4 is score no vdw ranking with ground truth only on clashing protein-ligand pairs,
    # index 5 is score no vdw ranking without ground truth only on clashing protein-ligand pairs
    # index 6 is glide ranking with ground truth only on clashing protein-ligand pairs,
    # index 7 is glide ranking without ground truth only on clashing protein-ligand pairs
    ls = []

    for i in range(len(LABELS)):
        ls.append([])

    score_no_vdw_data_without_correct = without_correct(score_no_vdw_data)
    glide_data_without_correct = without_correct(glide_data)

    for pair in score_no_vdw_data:
        # sort data in reverse ligand order (make sure that we don't choose ground truth in tie breakers)
        rev_score_no_vdw_data = sorted(score_no_vdw_data[pair], key=lambda x: x[0], reverse=True)
        rev_glide_data = sorted(glide_data[pair], key=lambda x: x[0], reverse=True)
        rev_score_no_vdw_data_without_correct = sorted(score_no_vdw_data_without_correct[pair], key=lambda x: x[0],
                                                        reverse=True)
        rev_glide_data_without_correct = sorted(glide_data_without_correct[pair], key=lambda x: x[0], reverse=True)
        sorted_score_no_vdw = sorted(rev_score_no_vdw_data, key=lambda x: x[1])
        sorted_glide = sorted(rev_glide_data, key=lambda x: x[1])
        sorted_score_no_vdw_without_correct = sorted(rev_score_no_vdw_data_without_correct, key=lambda x: x[1])
        sorted_glide_without_correct = sorted(rev_glide_data_without_correct, key=lambda x: x[1])
        target = pair.split('-')[0]
        num_glide = find_num_glide(max_poses, sorted_score_no_vdw, target)
        num_poses = min(num_glide, max_poses)
        ls[0].append(min(sorted_score_no_vdw[:num_poses], key=lambda x: x[2])[2])
        ls[1].append(min(sorted_score_no_vdw_without_correct[:num_poses], key=lambda x: x[2])[2])
        ls[2].append(min(sorted_glide[:num_poses], key=lambda x: x[2])[2])
        ls[3].append(min(sorted_glide_without_correct[:num_poses], key=lambda x: x[2])[2])
        if pair in process:
            ls[4].append(min(sorted_score_no_vdw[:num_poses], key=lambda x: x[2])[2])
            ls[5].append(min(sorted_score_no_vdw_without_correct[:num_poses], key=lambda x: x[2])[2])
            ls[6].append(min(sorted_glide[:num_poses], key=lambda x: x[2])[2])
            ls[7].append(min(sorted_glide_without_correct[:num_poses], key=lambda x: x[2])[2])

    return ls

def bar_graph(freq_list, out_dir):
    x_labels = [i for i in range(len(freq_list[0]))]
    fig, ax = plt.subplots()
    for i in range(len(LABELS) // 2):
        plt.plot(x_labels, freq_list[i], label=LABELS[i])

    ax.legend()
    ax.set_xlabel('Pose Cutoff')
    ax.set_ylabel('Percent')
    plt.savefig(os.path.join(out_dir, 'score_no_vdw_accuracy.png'))

    fig, ax = plt.subplots()
    for i in range(len(LABELS) // 2, len(LABELS)):
        plt.plot(x_labels, freq_list[i], label=LABELS[i])

    ax.legend()
    ax.set_xlabel('Pose Cutoff')
    ax.set_ylabel('Percent')
    plt.savefig(os.path.join(out_dir, 'score_no_vdw_accuracy_clash.png'))


def run_all_data(docked_prot_file, clash_docked_prot_file, run_path, label_file, score_no_vdw_dir, glide_dir, graph_dir,
                 grouped_files, n):
    for i, group in enumerate(grouped_files):
        cmd = 'sbatch -p owners -t 1:00:00 -o {} --wrap="' \
              '/home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python score_no_vdw_analysis.py ' \
              'group_data {} {} {} {} {} {} {} --n {} --index {}"'
        os.system(cmd.format(os.path.join(run_path, 'graph_{}.out'.format(i)), docked_prot_file, clash_docked_prot_file,
                             run_path, label_file, score_no_vdw_dir, glide_dir, graph_dir, n, i))

def run_group_data(grouped_files, label_df, index, score_no_vdw_dir, glide_dir):
    score_no_vdw_data = {}
    glide_data = {}
    for protein, target, start in grouped_files[index]:
        pair = '{}-to-{}'.format(target, start)
        print(pair)
        if target not in score_no_vdw_data:
            score_no_vdw_data[pair] = []
            glide_data[pair] = []
        for i in range(100):
            pdb_code = '{}_lig{}'.format(target, i)
            if len(label_df[label_df['target'] == pdb_code]) != 0:
                rmsd = get_label(pdb_code, label_df)
                # insert pdb code (index 0), score no vdw (index 1), rmsd (index 2)
                score_no_vdw_data[pair].append((pdb_code, get_score_no_vdw(pdb_code, label_df), rmsd))
                glide_data[pair].append((pdb_code, get_glide_score(pdb_code, label_df), rmsd))
        print(len(glide_data[pair]))

    print(len(glide_data))
    # outfile = open(os.path.join(score_no_vdw_dir, '{}.pkl'.format(index)), 'wb')
    # pickle.dump(score_no_vdw_data, outfile)
    outfile = open(os.path.join(glide_dir, '{}.pkl'.format(index)), 'wb')
    pickle.dump(glide_data, outfile)

def run_check_data(grouped_files, score_no_vdw_dir, glide_dir):
    unfinished = []
    for i in range(len(grouped_files)):
        if not os.path.exists(os.path.join(score_no_vdw_dir, '{}.pkl'.format(i))):
            unfinished.append(i)

    print('Missing from score no vdw:', len(unfinished), '/', len(grouped_files))
    print(unfinished)

    unfinished = []
    for i in range(len(grouped_files)):
        if not os.path.exists(os.path.join(glide_dir, '{}.pkl'.format(i))):
            unfinished.append(i)

    print('Missing from glide:', len(unfinished), '/', len(grouped_files))
    print(unfinished)

def get_prots(docked_prot_file):
    """
    gets list of all protein, target ligands, and starting ligands in the index file
    :param docked_prot_file: (string) file listing proteins to process
    :return: process (list) list of all protein, target ligands, and starting ligands to process
    """
    process = []
    with open(docked_prot_file) as fp:
        for line in tqdm(fp, desc='index file'):
            if line[0] == '#': continue
            protein, target, start = line.strip().split()
            process.append((protein, target, start))

    return process

def get_pairs(docked_prot_file):
    """
    gets list of all protein, target ligands, and starting ligands in the index file
    :param docked_prot_file: (string) file listing proteins to process
    :return: process (list) list of all protein, target ligands, and starting ligands to process
    """
    process = []
    with open(docked_prot_file) as fp:
        for line in tqdm(fp, desc='index file'):
            if line[0] == '#': continue
            protein, target, start = line.strip().split()
            process.append('{}-to-{}'.format(target, start))

    return process

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either all, group, check, '
                                               'all_dist_check, group_dist_check, check_dist_check, '
                                               'all_name_check, group_name_check, check_name_check, or delete')
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('clash_docked_prot_file', type=str, help='file listing clashing proteins')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('label_file', type=str, help='file with rmsd labels')
    parser.add_argument('score_no_vdw_dir', type=str, help='directory where all score no vdw data will be saved')
    parser.add_argument('glide_dir', type=str, help='directory where all glide data will be saved')
    parser.add_argument('graph_dir', type=str, help='directory where all graphs will be saved')
    parser.add_argument('--index', type=int, default=-1, help='for group task, group number')
    parser.add_argument('--n', type=int, default=3, help='number of protein, target, start groups processed in '
                                                         'group task')
    parser.add_argument('--cutoff', type=int, default=2, help='rmsd accuracy cutoff between predicted ligand pose and '
                                                              'true ligand pose')
    parser.add_argument('--max_poses', type=int, default=100, help='maximum number of glide poses considered')
    parser.add_argument('--clash_docked_prot_file', type=str, help='file listing clashing proteins')
    args = parser.parse_args()

    if not os.path.exists(args.run_path):
        os.mkdir(args.run_path)

    if not os.path.exists(args.score_no_vdw_dir):
        os.mkdir(args.score_no_vdw_dir)

    if not os.path.exists(args.glide_dir):
        os.mkdir(args.glide_dir)

    label_df = pd.read_csv(args.label_file)

    if args.task == 'all_data':
        process = get_prots(args.docked_prot_file)
        grouped_files = group_files(args.n, process)
        run_all_data(args.docked_prot_file, args.clash_docked_prot_file, args.run_path, args.label_file,
                     args.score_no_vdw_dir, args.glide_dir, args.graph_dir, grouped_files, args.n)

    elif args.task == 'group_data':
        process = get_prots(args.docked_prot_file)
        grouped_files = group_files(args.n, process)
        run_group_data(grouped_files, label_df, args.index, args.score_no_vdw_dir, args.glide_dir)

    elif args.task == 'check_data':
        process = get_prots(args.docked_prot_file)
        grouped_files = group_files(args.n, process)
        run_check_data(grouped_files, args.score_no_vdw_dir, args.glide_dir)

    elif args.task == 'analyze':
        process = get_pairs(args.clash_docked_prot_file)
        score_no_vdw_data = get_data(args.score_no_vdw_dir)
        glide_data = get_data(args.glide_dir)
        print(len(score_no_vdw_data))

        score_list = []
        for i in tqdm(range(1, args.max_poses), desc='iterating from 1 to 100 top poses'):
            score_list.append(get_graph_data(score_no_vdw_data, glide_data, process, i))

        freq_list = []
        for i in range(len(score_list[0])):
            accuracy_ls = []
            for ls in score_list:
                accuracy_ls.append(len([j for j in ls[i] if j < args.cutoff]) * 100 / len(ls[i]))
            freq_list.append(accuracy_ls)

        bar_graph(freq_list, args.graph_dir)


if __name__=="__main__":
    main()