"""
The purpose of this code is to get the physics scores and the rmsds

It can be run on sherlock using
$ $SCHRODINGER/run python3 score_and_rmsd.py run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
$ $SCHRODINGER/run python3 score_and_rmsd.py check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
$ $SCHRODINGER/run python3 score_and_rmsd.py delete_json /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
"""

import argparse
import os
from tqdm import tqdm

import sys
sys.path.append('/home/users/sidhikab/docking')
from docking.docking_class import Docking_Set

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

def run(process, run_path, raw_root, n, max_num_concurrent_jobs):
    """
    get scores and rmsds
    :param process: (list) list of all protein, target, start
    :param run_path: (string) directory where script and output files will be written
    :param raw_root: (string) directory where raw data will be placed
    :param n: (int) number of protein, target, start groups processed in group task
    :return:
    """
    docking_config = []
    for protein, target, start in process:
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, pair)
        pose_path = os.path.join(pair_path, 'ligand_poses')
        docking_config.append({'folder': pair_path,
                               'name': pair,
                               'grid_file': os.path.join(pair_path, '{}.zip'.format(pair)),
                               'prepped_ligand_file': os.path.join(pair_path, '{}_merge_pv.mae'.format(pair)),
                               'glide_settings': {'num_poses': 1, 'docking_method': 'inplace'},
                               'ligand_file': os.path.join(pose_path, '{}_lig0.mae'.format(target))})
        if len(docking_config) == max_num_concurrent_jobs:
            break

    print(len(docking_config))

    run_config = {'run_folder': run_path,
                  'group_size': n,
                  'partition': 'owners',
                  'dry_run': False}

    dock_set = Docking_Set()
    dock_set.run_docking_rmsd_delete(docking_config, run_config)

def check(docked_prot_file, raw_root):
    """
    check if scores and rmsds were calculated
    :param docked_prot_file: (string) file listing proteins to process
    :param raw_root: (string) directory where raw data will be placed
    :return:
    """
    counter = 0
    unfinished = []
    incomplete = []
    with open(docked_prot_file) as fp:
        for line in tqdm(fp, desc='protein, target, start groups'):
            if line[0] == '#': continue
            protein, target, start = line.strip().split()
            pair = '{}-to-{}'.format(target, start)
            counter += 1
            docking_config = []
            protein_path = os.path.join(raw_root, protein)
            pair_path = os.path.join(protein_path, pair)
            pose_path = os.path.join(pair_path, 'ligand_poses')
            docking_config.append({'folder': pair_path,
                                   'name': pair,
                                   'grid_file': os.path.join(pair_path, '{}.zip'.format(pair)),
                                   'prepped_ligand_file': os.path.join(pair_path, '{}_merge_pv.mae.gz'.format(pair)),
                                   'glide_settings': {'num_poses': 1, 'docking_method': 'inplace'},
                                   'ligand_file': os.path.join(pose_path, '{}_lig0.mae'.format(target))})

            dock_set = Docking_Set()
            if not os.path.exists(os.path.join(pair_path, '{}.scor'.format(pair))):
                unfinished.append((protein, target, start))
            else:
                if not os.path.exists(os.path.join(pair_path, '{}_rmsd.csv'.format(pair))):
                    print(os.path.join(pair_path, '{}_rmsd.csv'.format(pair)))
                    unfinished.append((protein, target, start))
                results = dock_set.get_docking_gscores(docking_config, mode='multi')
                results_by_ligand = results[pair]
                if len(results_by_ligand.keys()) != len(os.listdir(pose_path)):
                    print(len(results_by_ligand.keys()), len(os.listdir(pose_path)))
                    incomplete.append((protein, target, start))

        print('Missing', len(unfinished), '/', counter)
        print('Incomplete', len(incomplete), '/', counter - len(unfinished))
        print(incomplete)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either run, check, or delete_json')
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('--index', type=int, default=-1, help='for group task, group number')
    parser.add_argument('--n', type=int, default=3, help='number of protein, target, start groups processed in '
                                                         'group task')
    parser.add_argument('--max_num_concurrent_jobs', type=int, default=200, help='maximum number of concurrent jobs '
                                                                                 'that can be run on slurm at one time')
    args = parser.parse_args()

    if not os.path.exists(args.run_path):
        os.mkdir(args.run_path)

    if args.task == 'run':
        process = []
        with open(args.docked_prot_file) as fp:
            for line in tqdm(fp, desc='protein, target, start groups'):
                if line[0] == '#': continue
                protein, target, start = line.strip().split()
                pair = '{}-to-{}'.format(target, start)
                protein_path = os.path.join(args.raw_root, protein)
                pair_path = os.path.join(protein_path, pair)

                if not os.path.exists(os.path.join(pair_path, '{}.scor'.format(pair))):
                    process.append((protein, target, start))
        run(process, args.run_path, args.raw_root, args.n)

    if args.task == 'check':
        check(args.docked_prot_file, args.raw_root)

    if args.task == 'delete_json':
        with open(args.docked_prot_file) as fp:
            for line in tqdm(fp, desc='protein, target, start groups'):
                if line[0] == '#': continue
                protein, target, start = line.strip().split()
                pair = '{}-to-{}'.format(target, start)
                protein_path = os.path.join(args.raw_root, protein)
                pair_path = os.path.join(protein_path, pair)
                if os.path.exists(os.path.join(pair_path, '{}_state.json'.format(pair))):
                    os.remove(os.path.join(pair_path, '{}_state.json'.format(pair)))

if __name__=="__main__":
    main()