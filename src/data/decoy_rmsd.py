"""
The purpose of this code is to obtain the rmsd between each ligand pose and the ground truth ligand pose

It can be run on sherlock using
ml load chemistry
ml load schrodinger
/home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python decoy_rmsd.py run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
/home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python decoy_rmsd.py check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
/home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python decoy_rmsd.py MAPK14 /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw

"""

import os
import argparse
from tqdm import tqdm
import pickle

PROTEIN_CUTOFF = 2000
N = 3

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either run, check, or MAPK14')
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('--index', type=int, default=-1, help='for group task, group number')
    args = parser.parse_args()

    if not os.path.exists(args.run_path):
        os.mkdir(args.run_path)

    if args.task == 'run':
        process = get_prots(args.docked_prot_file)
        grouped_files = group_files(N, process)

        for i in range(len(grouped_files)):
            with open(os.path.join(args.run_path, 'rmsd{}_in.sh'.format(i)), 'w') as f:
                f.write('#!/bin/bash\n')
                for protein, target, start in grouped_files[i]:
                    protein_path = os.path.join(args.raw_root, protein)
                    pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
                    pose_path = os.path.join(pair_path, 'ligand_poses')
                    index = []
                    files = []
                    for file in os.listdir(pose_path):
                        if '{}_lig'.format(target) in file and file[-3:] == 'mae':
                            index.append(file)
                            files.append(os.path.join(pose_path, file))
                    f.write('cat {}/{}_prot.mae {} > {}/{}-to-{}_merge_pv.mae\n'.format(
                        pair_path, start, ' '.join(files), pair_path, target, start))
                    f.write('$SCHRODINGER/run rmsd.py -use_neutral_scaffold -pv second -c {}/{}-to-{}_rmsd.out '
                            '{}/{}_lig0.mae {}/{}-to-{}_merge_pv.mae\n'.format(pair_path, target, start, pose_path,
                                                                               target, pair_path, target, start))
                    f.write('rm {}/{}-to-{}_merge_pv.mae\n'.format(pair_path, target, start))
                    with open(os.path.join(pair_path, '{}-to-{}_rmsd_index.pkl'.format(target, start)), 'wb') as pickle_f:
                        pickle.dump(index, pickle_f)
            os.chdir(args.run_path)
            os.system('sbatch -p owners -t 02:00:00 -o rmsd{}.out rmsd{}_in.sh'.format(i, i))

        print(len(grouped_files))

    if args.task == 'check':
        process = []
        num_pairs = 0
        with open(args.docked_prot_file) as fp:
            for line in tqdm(fp, desc='going through protein, target, start groups'):
                if line[0] == '#': continue
                protein, target, start = line.strip().split()
                num_pairs += 1
                protein_path = os.path.join(args.raw_root, protein)
                pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
                if not os.path.exists(os.path.join(pair_path, '{}-to-{}_rmsd.out'.format(target, start))):
                    process.append((protein, target, start))

        print('Missing', len(process), '/', num_pairs)
        print(process)

    if args.task == 'MAPK14':
        protein = 'MAPK14'
        ligs = ['3D83', '4F9Y']
        with open(os.path.join(args.run_path, 'rmsd_in.sh'), 'w') as f:
            f.write('#!/bin/bash\n')
            for target in ligs:
                for start in ligs:
                    if target != start:
                        pair_path = os.path.join(args.datapath, '{}/{}-to-{}'.format(protein, target, start))
                        f.write('cd {}\n'.format(pair_path))
                        files = []
                        for file in os.listdir(os.path.join(args.datapath, pair_path)):
                            if '{}_lig'.format(target) in file and file[-3:] == 'mae':
                                files.append(file)
                        f.write('cat {}_prot.mae {} > {}-to-{}_merge_pv.mae\n'.format(start, ' '.join(files), target,
                                                                                      start))
                        f.write(
                            '$SCHRODINGER/run rmsd.py -use_neutral_scaffold -pv second -c {}-to-{}_rmsd.out {}_lig0.mae '
                            '{}-to-{}_merge_pv.mae\n'.format(
                                target, start, target, target, start))
                        with open(os.path.join(pair_path, '{}-to-{}_rmsd_index.pkl'.format(target, start)),
                                  'wb') as pickle_f:
                            pickle.dump(files, pickle_f)

        os.chdir(args.run_path)
        os.system('sbatch -p rondror -t 02:00:00 -o rmsd.out rmsd_in.sh')

if __name__ == "__main__":
    main()