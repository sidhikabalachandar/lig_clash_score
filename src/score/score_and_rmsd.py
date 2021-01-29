"""
The purpose of this code is to get the physics scores and the rmsds

It can be run on sherlock using
$ $SCHRODINGER/run python3 score_and_rmsd.py run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw

$ $SCHRODINGER/run python3 score_and_rmsd.py check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits/combined_index_balance_clash_large.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw  --decoy_type conformer_poses

$ $SCHRODINGER/run python3 score_and_rmsd.py check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
$ $SCHRODINGER/run python3 score_and_rmsd.py delete /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
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
        for line in tqdm(fp, desc='protein, target, start groups'):
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

def run(process, run_path, raw_root, decoy_type, n, max_num_concurrent_jobs):
    """
    get scores and rmsds
    :param process: (list) list of all protein, target, start
    :param run_path: (string) directory where script and output files will be written
    :param raw_root: (string) directory where raw data will be placed
    :param n: (int) number of protein, target, start groups processed in group task
    :return:
    """
    docking_config = []
    print(len(process))
    for protein, target, start in process:
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, pair)
        pose_path = os.path.join(pair_path, decoy_type)
        if not os.path.exists(os.path.join(pair_path, '{}_{}.scor'.format(pair, decoy_type))):
            docking_config.append({'folder': pair_path,
                               'name': '{}_{}'.format(pair, decoy_type),
                               'grid_file': os.path.join(pair_path, '{}.zip'.format(pair)),
                               'prepped_ligand_file': os.path.join(pair_path, '{}_{}_merge_pv.mae'.format(pair,
                                                                                                          decoy_type)),
                               'glide_settings': {'num_poses': 1, 'docking_method': 'inplace'},
                               'ligand_file': os.path.join(pose_path, '{}_lig0.mae'.format(target))})
        if len(docking_config) == max_num_concurrent_jobs:
            break
    print(len(docking_config))

    run_config = {'run_folder': run_path,
                  'group_size': n,
                  'partition': 'rondror',
                  'dry_run': False}

    dock_set = Docking_Set()
    dock_set.run_docking_rmsd_delete(docking_config, run_config)

def check(docked_prot_file, raw_root, decoy_type):
    """
    check if scores and rmsds were calculated
    :param docked_prot_file: (string) file listing proteins to process
    :param raw_root: (string) directory where raw data will be placed
    :return:
    """
    counter = 0
    missing = []
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
            pose_path = os.path.join(pair_path, decoy_type)
            docking_config.append({'folder': pair_path,
                                   'name': '{}_{}'.format(pair, decoy_type),
                                   'grid_file': os.path.join(pair_path, '{}.zip'.format(pair)),
                                   'prepped_ligand_file': os.path.join(pair_path, '{}_{}_merge_pv.mae'.format(pair,
                                                                                                              decoy_type)),
                                   'glide_settings': {'num_poses': 1, 'docking_method': 'inplace'},
                                   'ligand_file': os.path.join(pose_path, '{}_lig0.mae'.format(target))})

            dock_set = Docking_Set()
            if not os.path.exists(os.path.join(pair_path, '{}_{}.scor'.format(pair, decoy_type))):
                print(os.path.join(pair_path, '{}_{}.scor'.format(pair, decoy_type)))
                missing.append((protein, target, start))
                continue
            else:
                if not os.path.exists(os.path.join(pair_path, '{}_{}_rmsd.csv'.format(pair, decoy_type))):
                    print(os.path.join(pair_path, '{}_{}_rmsd.csv'.format(pair, decoy_type)))
                    incomplete.append((protein, target, start))
                    continue
                results = dock_set.get_docking_gscores(docking_config, mode='multi')
                results_by_ligand = results['{}_{}'.format(pair, decoy_type)]
                if len(results_by_ligand.keys()) != 100:
                    # print(results_by_ligand.keys())
                    print(len(results_by_ligand.keys()), 100)
                    incomplete.append((protein, target, start))
                    continue

        print('Missing', len(missing), '/', counter)
        print('Incomplete', len(incomplete), '/', counter - len(missing))
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
    parser.add_argument('--decoy_type', type=str, default='ligand_poses', help='either cartesian_poses, ligand_poses, '
                                                                               'or conformer_poses')
    args = parser.parse_args()

    if not os.path.exists(args.run_path):
        os.mkdir(args.run_path)

    if args.task == 'run':
        process = get_prots(args.docked_prot_file)
        run(process, args.run_path, args.raw_root, args.decoy_type, args.n, args.max_num_concurrent_jobs)

    if args.task == 'check':
        check(args.docked_prot_file, args.raw_root, args.decoy_type)

    if args.task == 'delete':
        with open(args.docked_prot_file) as fp:
            for line in tqdm(fp, desc='protein, target, start groups'):
                if line[0] == '#': continue
                protein, target, start = line.strip().split()
                pair = '{}-to-{}'.format(target, start)
                target_pair = '{}-to-{}'.format(target, target)
                protein_path = os.path.join(args.raw_root, protein)
                pair_path = os.path.join(protein_path, pair)
                # if os.path.exists(os.path.join(pair_path, '{}_conformer_poses.in'.format(pair))):
                #     os.remove(os.path.join(pair_path, '{}_conformer_poses.in'.format(pair)))
                # if os.path.exists(os.path.join(pair_path, '{}_conformer_poses.log'.format(pair))):
                #     os.remove(os.path.join(pair_path, '{}_conformer_poses.log'.format(pair)))
                # if os.path.exists(os.path.join(pair_path, '{}_conformer_poses_rmsd.csv'.format(pair))):
                #     os.remove(os.path.join(pair_path, '{}_conformer_poses_rmsd.csv'.format(pair)))
                # if os.path.exists(os.path.join(pair_path, '{}_conformer_poses.scor'.format(pair))):
                #     os.remove(os.path.join(pair_path, '{}_conformer_poses.scor'.format(pair)))

                if os.path.exists(os.path.join(pair_path, '{}_conformer_poses.in'.format(target_pair))):
                    os.remove(os.path.join(pair_path, '{}_conformer_poses.in'.format(target_pair)))
                if os.path.exists(os.path.join(pair_path, '{}_conformer_poses.log'.format(target_pair))):
                    os.remove(os.path.join(pair_path, '{}_conformer_poses.log'.format(target_pair)))
                if os.path.exists(os.path.join(pair_path, '{}_conformer_poses_rmsd.csv'.format(target_pair))):
                    os.remove(os.path.join(pair_path, '{}_conformer_poses_rmsd.csv'.format(target_pair)))
                if os.path.exists(os.path.join(pair_path, '{}_conformer_poses.scor'.format(target_pair))):
                    os.remove(os.path.join(pair_path, '{}_conformer_poses.scor'.format(target_pair)))
                if os.path.exists(os.path.join(pair_path, '{}_conformer_poses_merge_pv.mae'.format(target_pair))):
                    os.remove(os.path.join(pair_path, '{}_conformer_poses_merge_pv.mae'.format(target_pair)))
                # if os.path.exists(os.path.join(pair_path, '{}_conformer_poses_merge_pv.mae.gz'.format(pair))):
                #     os.remove(os.path.join(pair_path, '{}_conformer_poses_merge_pv.mae.gz'.format(pair)))
                # if os.path.exists(os.path.join(pair_path, '{}_conformer_poses_pv.maegz'.format(pair))):
                #     os.remove(os.path.join(pair_path, '{}_conformer_poses_pv.maegz'.format(pair)))
                # if os.path.exists(os.path.join(pair_path, '{}_conformer_poses_pv.maegz'.format(target_pair))):
                #     os.remove(os.path.join(pair_path, '{}_conformer_poses_pv.maegz'.format(target_pair)))
                # if os.path.exists(os.path.join(pair_path, 'aligned_conformers.mae')):
                #     os.remove(os.path.join(pair_path, 'aligned_conformers.mae'))
                # if os.path.exists(os.path.join(pair_path, '{}_lig.log'.format(start))):
                #     os.remove(os.path.join(pair_path, '{}_lig.log'.format(start)))
                # if os.path.exists(os.path.join(pair_path, '{}_lig-out.maegz'.format(start))):
                #     os.remove(os.path.join(pair_path, '{}_lig-out.maegz'.format(start)))

if __name__=="__main__":
    main()