"""
The purpose of this code is to create the split files

It can be run on sherlock using
$ $SCHRODINGER/run python3 cluster.py all /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
$ $SCHRODINGER/run python3 cluster.py group /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw --index 0
$ $SCHRODINGER/run python3 cluster.py check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
"""

import argparse
import os
import random
import pickle
from tqdm import tqdm
import schrodinger.structure as structure
from schrodinger.structutils.transform import get_centroid
from sklearn.cluster import KMeans
import numpy as np

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

def run_all(docked_prot_file, run_path, raw_root, grouped_files, n):
    """
    submits sbatch script to create decoys for each protein, target, start group
    :param docked_prot_file: (string) file listing proteins to process
    :param run_path: (string) directory where script and output files will be written
    :param raw_root: (string) directory where raw data will be placed
    :param grouped_files: (list) list of protein, target, start groups
    :return:
    """
    for i, group in enumerate(grouped_files):
        cmd = 'sbatch -p owners -t 1:00:00 -o {} --wrap="$SCHRODINGER/run python3 cluster.py group {} {} {} --n {} ' \
              '--index {}"'
        os.system(cmd.format(os.path.join(run_path, 'decoy{}.out'.format(i)), docked_prot_file,
                             run_path, raw_root, n, i))

def run_group(grouped_files, raw_root, index, num_clusters):
    for protein, target, start in grouped_files[index]:
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, pair)
        pose_path = os.path.join(pair_path, 'ligand_poses')
        graph_dir = '{}/{}-to-{}_graph.pkl'.format(pair_path, target, start)
        infile = open(graph_dir, 'rb')
        graph_data = pickle.load(infile)
        infile.close()
        centroids = []
        codes_to_include = []
        for i, pdb_code in tqdm(enumerate(graph_data), desc="pdb_codes"):
            if pdb_code.split('_')[-1][:4] != 'lig0' and pdb_code[-1].isalpha():
                file = os.path.join(pose_path, '{}.mae'.format(pdb_code))
                s = list(structure.StructureReader(file))[0]
                centroids.append((get_centroid(s), pdb_code))
            else:
                codes_to_include.append(pdb_code)
        if len(centroids) > num_clusters:
            X = np.zeros((len(centroids), 3))
            for i in range(len(X)):
                X[i] = centroids[i][0][:3]

            kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X)
            condensed = {}
            for i, label in enumerate(kmeans.labels_):
                if len(condensed) == num_clusters:
                    break
                if label not in condensed:
                    condensed[label] = centroids[i][1]
            codes_to_include.extend(list(condensed.values()))
        else:
            codes_to_include = list(graph_data.keys())

        outfile = open(os.path.join(pair_path, '{}_clustered.pkl'.format(pair)), 'wb')
        pickle.dump(codes_to_include, outfile)

def run_check(docked_prot_file, raw_root):
    """
    check if all files are created
    :param docked_prot_file: (string) file listing proteins to process
    :param raw_root: (string) directory where raw data will be placed
    :param max_poses: (int) maximum number of glide poses considered
    :param max_decoys: (int) maximum number of decoys created per glide pose
    :return:
    """
    process = []
    num_pairs = 0
    with open(docked_prot_file) as fp:
        for line in tqdm(fp, desc='protein, target, start groups'):
            if line[0] == '#': continue
            protein, target, start = line.strip().split()
            pair = '{}-to-{}'.format(target, start)
            num_pairs += 1
            protein_path = os.path.join(raw_root, protein)
            pair_path = os.path.join(protein_path, pair)
            if not os.path.exists(os.path.join(pair_path, '{}_clustered.pkl'.format(pair))):
                process.append((protein, target, start))

    print('Missing', len(process), '/', num_pairs)
    print(process)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either all, group, check, '
                                               'all_dist_check, group_dist_check, check_dist_check, '
                                               'all_name_check, group_name_check, check_name_check, or delete')
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('--n', type=int, default=3, help='number of protein, target, start groups processed in '
                                                         'group task')
    parser.add_argument('--num_clusters', type=int, default=100, help='number of clusters')
    parser.add_argument('--index', type=int, default=-1, help='for group task, group number')
    args = parser.parse_args()

    random.seed(0)
    if not os.path.exists(args.run_path):
        os.mkdir(args.run_path)

    if args.task == 'all':
        process = get_prots(args.docked_prot_file)
        grouped_files = group_files(args.n, process)
        run_all(args.docked_prot_file, args.run_path, args.raw_root, grouped_files, args.n)

    if args.task == 'group':
        process = get_prots(args.docked_prot_file)
        grouped_files = group_files(args.n, process)
        run_group(grouped_files, args.raw_root, args.index, args.num_clusters)

    if args.task == 'check':
        run_check(args.docked_prot_file, args.raw_root)

if __name__=="__main__":
    main()