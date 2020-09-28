"""
The purpose of this code is to set the train, val, and test data sets
It can be run on sherlock using

ml load chemistry
ml load schrodinger
$SCHRODINGER/run python3 clash_analysis.py all /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /home/users/sidhikab/lig_clash_score/reports/figures /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/combined.csv
$SCHRODINGER/run python3 clash_analysis.py group /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /home/users/sidhikab/lig_clash_score/reports/figures /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/combined.csv
$SCHRODINGER/run python3 clash_analysis.py check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /home/users/sidhikab/lig_clash_score/reports/figures /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/combined.csv
$SCHRODINGER/run python3 clash_analysis.py analyze /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /home/users/sidhikab/lig_clash_score/reports/figures /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/combined.csv
$SCHRODINGER/run python3 clash_analysis.py all_glide_data /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /home/users/sidhikab/lig_clash_score/reports/figures /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/combined.csv
$SCHRODINGER/run python3 clash_analysis.py group_glide_data /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /home/users/sidhikab/lig_clash_score/reports/figures /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/combined.csv --index 0
$SCHRODINGER/run python3 clash_analysis.py check_glide_data /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /home/users/sidhikab/lig_clash_score/reports/figures /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/combined.csv
$SCHRODINGER/run python3 clash_analysis.py graph /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /home/users/sidhikab/lig_clash_score/reports/figures /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/combined.csv
$SCHRODINGER/run python3 clash_analysis.py index /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /home/users/sidhikab/lig_clash_score/reports/figures /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/combined.csv --clash_prot_file /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random_clash.txt --non_clash_prot_file /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random_non_clash.txt
"""

import argparse
import os
import schrodinger.structutils.interactions.steric_clash as steric_clash
from schrodinger.structure import StructureReader
import pickle
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

def get_label(pdb, label_df):
    """
    searches for pdb's rmsd in combined rmsd df
    :param pdb: (string) {target}_lig{id}
    :param label_df: (df) combined rmsd df
    :return: (float) rmsd value
    """
    return label_df[label_df['target'] == pdb]['rmsd'].iloc[0]

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

def get_glide_data(glide_dir):
    glide_data = {}
    for file in os.listdir(glide_dir):
        infile = open(os.path.join(glide_dir, file), 'rb')
        group_glide_dict = pickle.load(infile)
        infile.close()
        glide_data.update(group_glide_dict)

    return glide_data

def run_all(docked_prot_file, run_path, raw_root, save_path, label_file, clash_n, grouped_files):
    """
    submits sbatch script to check mean distance of displacement for decoys for each protein, target, start group
    :param docked_prot_file: (string) file listing proteins to process
    :param run_path: (string) directory where script and output files will be written
    :param raw_root: (string) directory where raw data will be placed
    :param data_root: (string) pdbbind directory where raw data will be obtained
    :param grouped_files: (list) list of protein, target, start groups
    :return:
    """
    for i, group in enumerate(grouped_files):
        cmd = 'sbatch -p owners -t 1:00:00 -o {} --wrap="$SCHRODINGER/run python3 clash_analysis.py group {} {} {} {} {} ' \
              '--clash_n {} --index {}"'
        os.system(cmd.format(os.path.join(run_path, 'clash{}.out'.format(i)), docked_prot_file,
                             run_path, raw_root, save_path, label_file, clash_n, i))

def run_group(grouped_files, raw_root, index, clash_dir):
    """
    checks mean distance of displacement for decoys for each protein, target, start group
    :param grouped_files: (list) list of protein, target, start groups
    :param raw_root: (string) directory where raw data will be placed
    :param index: (int) group number
    :param dist_dir: (string) directiory to place distances
    :param max_poses: (int) maximum number of glide poses considered
    :param max_decoys: (int) maximum number of decoys created per glide pose
    :return:
    """
    clash_dict = {}
    for protein, target, start in grouped_files[index]:
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
        pose_path = os.path.join(pair_path, 'ligand_poses')
        struct_path = os.path.join(pair_path, '{}_prot.mae'.format(start))
        lig_path = os.path.join(pose_path, '{}_lig0.mae'.format(target))
        s1 = list(StructureReader(struct_path))[0]
        lig = list(StructureReader(lig_path))[0]
        clash_dict[(protein, target, start)] = steric_clash.clash_volume(s1, struc2=lig)

    outfile = open(os.path.join(clash_dir, '{}.pkl'.format(index)), 'wb')
    pickle.dump(clash_dict, outfile)

def run_check(grouped_files, clash_dir):
    """
    check if all dist files created and if all means are appropriate
    :param grouped_files: (list) list of protein, target, start groups
    :param dist_dir: (string) directiory to place distances
    :return:
    """
    if len(os.listdir(clash_dir)) != len(grouped_files):
        print('Not all files created')
    else:
        print('All files created')

def run_all_glide_data(docked_prot_file, run_path, raw_root, save_path, label_file, glide_n, grouped_files):
    """
    submits sbatch script to check mean distance of displacement for decoys for each protein, target, start group
    :param docked_prot_file: (string) file listing proteins to process
    :param run_path: (string) directory where script and output files will be written
    :param raw_root: (string) directory where raw data will be placed
    :param data_root: (string) pdbbind directory where raw data will be obtained
    :param grouped_files: (list) list of protein, target, start groups
    :return:
    """
    for i, group in enumerate(grouped_files):
        cmd = 'sbatch -p owners -t 1:00:00 -o {} --wrap="$SCHRODINGER/run python3 clash_analysis.py group_glide_data ' \
              '{} {} {} {} {} --glide_n {} --index {}"'
        os.system(cmd.format(os.path.join(run_path, 'glide{}.out'.format(i)), docked_prot_file,
                             run_path, raw_root, save_path, label_file, glide_n, i))

def run_group_glide_data(grouped_files, label_df, index, glide_dir, max_poses):
    glide_data = {}
    for protein, target, start in tqdm(grouped_files[index], desc='gnn dict group'):
        if target not in glide_data:
            glide_data[(protein, target, start)] = []

        for i in range(1, max_poses):
            pdb_code = '{}_lig{}'.format(target, i)
            if len(label_df[label_df['target'] == pdb_code]) != 0:
                # insert pdb code (index 0), glide rank (index 1), rmsd (index 2)
                glide_data[(protein, target, start)].append((pdb_code, i, get_label(pdb_code, label_df)))

    outfile = open(os.path.join(glide_dir, '{}.pkl'.format(index)), 'wb')
    pickle.dump(glide_data, outfile)

def run_check_glide_data(grouped_files, glide_dir):
    unfinished = []
    for i in range(len(grouped_files)):
        if not os.path.exists(os.path.join(glide_dir, '{}.pkl'.format(i))):
            unfinished.append(i)

    print('Missing:', len(unfinished), '/', len(grouped_files))
    print(unfinished)

def run_analyze(combined, save_path):
    sort_clash = sorted(combined.items(), key=lambda x: x[1], reverse=True)

    ax = sns.distplot([i[1] for i in sort_clash])
    fig = ax.get_figure()
    fig.savefig(os.path.join(save_path, 'clash_analysis.png'))

    for i in range(5):
        print(sort_clash[i])

    print()

    for i in range(-1, -6, -1):
        print(sort_clash[i])

    print()
    found = 0

    for i in range(len(sort_clash)):
        if sort_clash[i][1] > 200:
            found += 1

    print('fraction over 200', found / len(sort_clash))
    print('num over 200', found)
    print('total', len(sort_clash))

    for i in range(len(sort_clash)):
        if sort_clash[i][1] > 40:
            found += 1

    print('fraction over 40', found / len(sort_clash))
    print('num over 40', found)
    print('total', len(sort_clash))

def run_graph(combined, glide_data, save_path, cutoff):
    sort_clash = sorted(combined.items(), key=lambda x: x[1])

    clash_ordered_accuracies = {}
    ordered_pairs = []

    for group, clash in sort_clash:
        clash_ordered_accuracies[group] = min(glide_data[group], key=lambda x: x[2])[2]
        accuracy_ls = [1 for x in clash_ordered_accuracies if clash_ordered_accuracies[x] < cutoff]
        ordered_pairs.append((clash, sum(accuracy_ls) / len(clash_ordered_accuracies)))

    fig, ax = plt.subplots()
    ax.plot([x[0] for x in ordered_pairs], [x[1] for x in ordered_pairs])

    ax.set(xlabel='clash volume', ylabel='cumulative accuracy',
           title='clash volume vs cumulative accuracy')
    ax.grid()

    fig.savefig(os.path.join(save_path, 'clash_vs_accuracy.png'))

    sort_clash = sorted(combined.items(), key=lambda x: x[1], reverse=True)

    clash_ordered_accuracies = {}
    ordered_pairs = []

    for group, clash in sort_clash:
        clash_ordered_accuracies[group] = min(glide_data[group], key=lambda x: x[2])[2]
        accuracy_ls = [1 for x in clash_ordered_accuracies if clash_ordered_accuracies[x] < cutoff]
        ordered_pairs.append((clash, sum(accuracy_ls) / len(clash_ordered_accuracies)))

    fig, ax = plt.subplots()
    ax.plot([x[0] for x in ordered_pairs], [x[1] for x in ordered_pairs])

    ax.set(xlabel='clash volume', ylabel='cumulative accuracy',
           title='clash volume vs cumulative accuracy')
    ax.grid()

    fig.savefig(os.path.join(save_path, 'clash_vs_accuracy_reverse.png'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either all, group, check, analyze')
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('save_path', type=str, help='directory where graph will be placed')
    parser.add_argument('label_file', type=str, help='file with rmsd labels')
    parser.add_argument('--index', type=int, default=-1, help='for group task, group number')
    parser.add_argument('--clash_dir', type=str, default=os.path.join(os.getcwd(), 'clash'),
                        help='for all_dist_check and group_dist_check task, directiory to place distances')
    parser.add_argument('--clash_n', type=int, default=100, help='number of protein, target, start groups processed in '
                                                         'group task')
    parser.add_argument('--glide_n', type=int, default=3, help='number of protein, target, start groups processed in '
                                                                 'group_glide_data task')
    parser.add_argument('--glide_dir', type=str, default=os.path.join(os.getcwd(), 'glide'),
                        help='directory in which gnn data saved')
    parser.add_argument('--max_poses', type=int, default=100, help='maximum number of glide poses considered')
    parser.add_argument('--rmsd_cutoff', type=int, default=2, help='rmsd accuracy cutoff between predicted ligand pose and '
                                                              'true ligand pose')
    parser.add_argument('--clash_cutoff', type=int, default=40, help='rmsd accuracy cutoff between predicted ligand pose and '
                                                              'true ligand pose')
    parser.add_argument('--clash_prot_file', type=str, default=os.path.join(os.getcwd(), 'index.txt'),
                        help='for update task, name of prot file where clashing pairs will be placed')
    parser.add_argument('--non_clash_prot_file', type=str, default=os.path.join(os.getcwd(), 'index.txt'),
                        help='for update task, name of prot file where non-clashing pairs will be placed')
    args = parser.parse_args()

    if not os.path.exists(args.run_path):
        os.mkdir(args.run_path)

    if not os.path.exists(args.clash_dir):
        os.mkdir(args.clash_dir)

    if not os.path.exists(args.glide_dir):
        os.mkdir(args.glide_dir)

    label_df = pd.read_csv(args.label_file)

    if args.task == 'all':
        process = get_prots(args.docked_prot_file)
        grouped_files = group_files(args.n, process)
        run_all(args.docked_prot_file, args.run_path, args.raw_root, args.save_path, args.label_file, args.clash_n,
                grouped_files)

    elif args.task == 'group':
        process = get_prots(args.docked_prot_file)
        grouped_files = group_files(args.n, process)
        run_group(grouped_files, args.raw_root, args.index, args.clash_dir)

    elif args.task == 'check':
        process = get_prots(args.docked_prot_file)
        grouped_files = group_files(args.n, process)
        run_check(grouped_files, args.clash_dir)

    elif args.task == 'all_glide_data':
        combined = {}
        for file in os.listdir(args.clash_dir):
            infile = open(os.path.join(args.clash_dir, file), 'rb')
            clash_dict = pickle.load(infile)
            infile.close()
            combined.update(clash_dict)
        grouped_files = group_files(args.glide_n, list(combined.keys()))
        run_all_glide_data(args.docked_prot_file, args.run_path, args.raw_root, args.save_path, args.label_file,
                           args.glide_n, grouped_files)

    elif args.task == 'group_glide_data':
        combined = {}
        for file in os.listdir(args.clash_dir):
            infile = open(os.path.join(args.clash_dir, file), 'rb')
            clash_dict = pickle.load(infile)
            infile.close()
            combined.update(clash_dict)
        grouped_files = group_files(args.glide_n, list(combined.keys()))
        print(len(grouped_files))
        run_group_glide_data(grouped_files, label_df, args.index, args.glide_dir, args.max_poses)

    elif args.task == 'check_glide_data':
        combined = {}
        for file in os.listdir(args.clash_dir):
            infile = open(os.path.join(args.clash_dir, file), 'rb')
            clash_dict = pickle.load(infile)
            infile.close()
            combined.update(clash_dict)
        grouped_files = group_files(args.glide_n, list(combined.keys()))
        run_check_glide_data(grouped_files, args.glide_dir)

    if args.task == 'analyze':
        combined = {}
        for file in os.listdir(args.clash_dir):
            infile = open(os.path.join(args.clash_dir, file), 'rb')
            clash_dict = pickle.load(infile)
            infile.close()
            combined.update(clash_dict)
        run_analyze(combined, args.save_path)

    if args.task == 'graph':
        combined = {}
        for file in os.listdir(args.clash_dir):
            infile = open(os.path.join(args.clash_dir, file), 'rb')
            clash_dict = pickle.load(infile)
            infile.close()
            combined.update(clash_dict)

        glide_data = get_glide_data(args.glide_dir)
        run_graph(combined, glide_data, args.save_path, args.rmsd_cutoff)

    if args.task == 'index':
        combined = {}
        for file in os.listdir(args.clash_dir):
            infile = open(os.path.join(args.clash_dir, file), 'rb')
            clash_dict = pickle.load(infile)
            infile.close()
            combined.update(clash_dict)

        clash_text = []
        non_clash_text = []
        for protein, target, start in tqdm(combined, desc='protein, target, start groups'):
            if combined[(protein, target, start)] > args.clash_cutoff:
                clash_text.append('{} {} {}\n'.format(protein, target, start))
            else:
                non_clash_text.append('{} {} {}\n'.format(protein, target, start))

        file = open(args.clash_prot_file, "w")
        file.writelines(clash_text)
        file.close()
        file = open(args.non_clash_prot_file, "w")
        file.writelines(non_clash_text)
        file.close()

if __name__=="__main__":
    main()