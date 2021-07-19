"""
The purpose of this code is to get the physics scores and the rmsds

It can be run on sherlock using
$ $SCHRODINGER/run python3 score_and_rmsd.py add_data /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw --protein O38732 --target 2i0a --start 2q5k  --group_name exhaustive_grid_1_rotation_5
"""

import argparse
import os
from tqdm import tqdm

import sys
sys.path.append('/home/users/sidhikab/docking')
from docking.docking_class import Docking_Set
import schrodinger.structure as structure
import pandas as pd
from docking.utilities import score_no_vdW

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


def run(protein, target, start, run_path, raw_root, group_name, max_num_concurrent_jobs):
    """
    get scores and rmsds
    :param process: (list) list of all protein, target, start
    :param run_path: (string) directory where script and output files will be written
    :param raw_root: (string) directory where raw data will be placed
    :param n: (int) number of protein, target, start groups processed in group task
    :return:
    """
    docking_config = []
    pair = '{}-to-{}'.format(target, start)
    protein_path = os.path.join(raw_root, protein)
    pair_path = os.path.join(protein_path, pair)
    pose_path = os.path.join(pair_path, group_name)
    grouped_pose_path = os.path.join(pose_path, 'grouped_poses')
    dock_output_path = os.path.join(pose_path, 'dock_output')
    ground_truth_file = os.path.join(pair_path, 'ligand_poses', '{}_lig0.mae'.format(target))
    if not os.path.exists(dock_output_path):
        os.mkdir(dock_output_path)
    for file in os.listdir(grouped_pose_path):
        prefix = 'grouped_'
        suffix = '.maegz'
        name = file[len(prefix):-len(suffix)]
        if not os.path.exists(os.path.join(dock_output_path, '{}.scor'.format(name))):
            docking_config.append({'folder': dock_output_path,
                                   'name': name,
                                   'grid_file': os.path.join(pair_path, '{}.zip'.format(pair)),
                                   'prepped_ligand_file': os.path.join(grouped_pose_path, file),
                                   'glide_settings': {'num_poses': 1, 'docking_method': 'inplace'},
                                   'ligand_file': ground_truth_file})
        if len(docking_config) == max_num_concurrent_jobs:
            break
    print(len(docking_config))

    run_config = {'run_folder': run_path,
                  'group_size': 1,
                  'partition': 'rondror',
                  'dry_run': False}

    dock_set = Docking_Set()
    dock_set.run_docking_rmsd_delete(docking_config, run_config)


def check(raw_root, protein, target, start, group_name):
    """
    check if scores and rmsds were calculated
    :param docked_prot_file: (string) file listing proteins to process
    :param raw_root: (string) directory where raw data will be placed
    :return:
    """
    pair = '{}-to-{}'.format(target, start)
    protein_path = os.path.join(raw_root, protein)
    pair_path = os.path.join(protein_path, pair)
    pose_path = os.path.join(pair_path, group_name)
    grouped_pose_path = os.path.join(pose_path, 'grouped_poses')
    dock_output_path = os.path.join(pose_path, 'dock_output')
    ground_truth_file = os.path.join(pair_path, 'ligand_poses', '{}_lig0.mae'.format(target))
    missing = []
    incomplete = []
    if not os.path.exists(dock_output_path):
        os.mkdir(dock_output_path)
    for file in os.listdir(grouped_pose_path):
        prefix = 'grouped_'
        suffix = '.maegz'
        name = file[len(prefix):-len(suffix)]
        docking_config = [{'folder': dock_output_path,
                           'name': name,
                           'grid_file': os.path.join(pair_path, '{}.zip'.format(pair)),
                           'prepped_ligand_file': os.path.join(grouped_pose_path, file),
                           'glide_settings': {'num_poses': 1, 'docking_method': 'inplace'},
                           'ligand_file': ground_truth_file}]
        dock_set = Docking_Set()
        if not os.path.exists(os.path.join(dock_output_path, '{}.scor'.format(name))):
            if os.path.exists(os.path.join(dock_output_path, '{}.in'.format(name))):
                os.remove(os.path.join(dock_output_path, '{}.in'.format(name)))
            if os.path.exists(os.path.join(dock_output_path, '{}.log'.format(name))):
                os.remove(os.path.join(dock_output_path, '{}.log'.format(name)))
            if os.path.exists(os.path.join(dock_output_path, '{}_state.json'.format(name))):
                os.remove(os.path.join(dock_output_path, '{}_state.json'.format(name)))
            missing.append(name)
            continue
        else:
            if not os.path.exists(os.path.join(dock_output_path, '{}_rmsd.csv'.format(name))):
                print(os.path.join(dock_output_path, '{}_rmsd.csv'.format(name)))
                incomplete.append(name)
                continue
            results = dock_set.get_docking_gscores(docking_config, mode='multi')
            results_by_ligand = results[name]
            group = list(structure.StructureReader(os.path.join(grouped_pose_path, file)))
            if len(results_by_ligand.keys()) != len(group):
                print(len(results_by_ligand), len(group))
                incomplete.append(name)
                continue

    print('Missing', len(missing), '/', len(os.listdir(grouped_pose_path)))
    print('Incomplete', len(incomplete), '/', len(os.listdir(grouped_pose_path)) - len(missing))
    print(incomplete)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either run, check, or delete_json')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('--protein', type=str, default='', help='protein name')
    parser.add_argument('--target', type=str, default='', help='target ligand name')
    parser.add_argument('--start', type=str, default='', help='start ligand name')
    parser.add_argument('--index', type=int, default=-1, help='for group task, group number')
    parser.add_argument('--n', type=int, default=3, help='number of protein, target, start groups processed in '
                                                         'group task')
    parser.add_argument('--max_num_concurrent_jobs', type=int, default=200, help='maximum number of concurrent jobs '
                                                                                 'that can be run on slurm at one time')
    parser.add_argument('--group_name', type=str, default='', help='name of pose group subdir')
    args = parser.parse_args()

    if not os.path.exists(args.run_path):
        os.mkdir(args.run_path)

    if args.task == 'run':
        run(args.protein, args.target, args.start, args.run_path, args.raw_root, args.group_name,
            args.max_num_concurrent_jobs)

    elif args.task == 'check':
        check(args.raw_root, args.protein, args.target, args.start, args.group_name)

    elif args.task == 'add_data':
        pair = '{}-to-{}'.format(args.target, args.start)
        protein_path = os.path.join(args.raw_root, args.protein)
        pair_path = os.path.join(protein_path, pair)
        pose_path = os.path.join(pair_path, args.group_name)
        grouped_pose_path = os.path.join(pose_path, 'grouped_poses')
        dock_output_path = os.path.join(pose_path, 'dock_output')
        ground_truth_file = os.path.join(pair_path, 'ligand_poses', '{}_lig0.mae'.format(args.target))

        df = pd.read_csv(os.path.join(pair_path, 'poses_after_pred_filter.csv'))
        in_place_data = {}

        for file in tqdm(os.listdir(dock_output_path), desc='docked files'):
            if file[-4:] == 'scor':
                name = file[:-5]
                docking_config = [{'folder': dock_output_path,
                                   'name': name,
                                   'grid_file': os.path.join(pair_path, '{}.zip'.format(pair)),
                                   'prepped_ligand_file': os.path.join(grouped_pose_path, '{}.maegz'.format(name)),
                                   'glide_settings': {'num_poses': 1, 'docking_method': 'inplace'},
                                   'ligand_file': ground_truth_file}]
                dock_set = Docking_Set()
                results = dock_set.get_docking_gscores(docking_config, mode='multi')
                results_by_ligand = results[name]
                for n in results_by_ligand:
                    glide_score = results_by_ligand[n][0]['Score']
                    score = score_no_vdW(results_by_ligand[n][0])
                    modified_score_no_vdw = score
                    if score > 20:
                        modified_score_no_vdw = 20
                    elif score < -20:
                        modified_score_no_vdw = -20

                    in_place_data[n] = (glide_score, score, modified_score_no_vdw)

        glide_scores = []
        score_no_vdws = []
        modified_score_no_vdws = []
        for name in df['name'].to_list():
            glide_scores.append(in_place_data[name][0])
            score_no_vdws.append(in_place_data[name][1])
            modified_score_no_vdws.append(in_place_data[name][2])

        df.to_csv(os.path.join(pair_path, 'poses_after_pred_filter.csv'))

if __name__=="__main__":
    main()