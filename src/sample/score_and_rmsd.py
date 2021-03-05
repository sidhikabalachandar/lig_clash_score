"""
The purpose of this code is to get the physics scores and the rmsds

It can be run on sherlock using
$ $SCHRODINGER/run python3 score_and_rmsd.py add_glide /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw --protein O38732 --target 2i0a --start 2q5k  --max_num_concurrent_jobs 50
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


def run(protein, target, start, run_path, raw_root, decoy_type, max_num_concurrent_jobs):
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
    pose_path = os.path.join(pair_path, decoy_type)
    grouped_pose_path = os.path.join(pose_path, 'grouped_poses')
    dock_output_path = os.path.join(pose_path, 'dock_output')
    ground_truth_file = os.path.join(pair_path, 'ligand_poses', '{}_lig0.mae'.format(target))
    if not os.path.exists(dock_output_path):
        os.mkdir(dock_output_path)
    for file in os.listdir(grouped_pose_path):
        name = file[:-6]
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


def check(raw_root, decoy_type, protein, target, start):
    """
    check if scores and rmsds were calculated
    :param docked_prot_file: (string) file listing proteins to process
    :param raw_root: (string) directory where raw data will be placed
    :return:
    """
    pair = '{}-to-{}'.format(target, start)
    protein_path = os.path.join(raw_root, protein)
    pair_path = os.path.join(protein_path, pair)
    pose_path = os.path.join(pair_path, decoy_type)
    grouped_pose_path = os.path.join(pose_path, 'grouped_poses')
    dock_output_path = os.path.join(pose_path, 'dock_output')
    ground_truth_file = os.path.join(pair_path, 'ligand_poses', '{}_lig0.mae'.format(target))
    missing = []
    incomplete = []

    for file in os.listdir(grouped_pose_path):
        name = file[:-6]
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
    parser.add_argument('--decoy_type', type=str, default='grid_search_poses', help='either cartesian_poses, '
                                                                                    'ligand_poses, or conformer_poses')
    args = parser.parse_args()

    if not os.path.exists(args.run_path):
        os.mkdir(args.run_path)

    if args.task == 'run':
        run(args.protein, args.target, args.start, args.run_path, args.raw_root, args.decoy_type,
            args.max_num_concurrent_jobs)

    elif args.task == 'check':
        check(args.raw_root, args.decoy_type, args.protein, args.target, args.start)

    elif args.task == 'combine':
        pair = '{}-to-{}'.format(args.target, args.start)
        protein_path = os.path.join(args.raw_root, args.protein)
        pair_path = os.path.join(protein_path, pair)
        pose_path = os.path.join(pair_path, args.decoy_type)
        dfs = []
        for file in os.listdir(pose_path):
            if file[-9:] == 'poses.csv':
                df = pd.read_csv(os.path.join(pose_path, file))
                dfs.append(df)

        combined_df = pd.concat(dfs)
        combined_df.to_csv(os.path.join(pose_path, 'combined.csv'))

    elif args.task == 'add_data':
        pair = '{}-to-{}'.format(args.target, args.start)
        protein_path = os.path.join(args.raw_root, args.protein)
        pair_path = os.path.join(protein_path, pair)
        pose_path = os.path.join(pair_path, args.decoy_type)
        grouped_pose_path = os.path.join(pose_path, 'grouped_poses')
        dock_output_path = os.path.join(pose_path, 'dock_output')
        ground_truth_file = os.path.join(pair_path, 'ligand_poses', '{}_lig0.mae'.format(args.target))

        df = pd.read_csv(os.path.join(pose_path, 'combined.csv'))
        in_place_data = {'name': [], 'glide_score': [], 'score_no_vdw': [], 'modified_score_no_vdw': [],
                         'start_clash': [], 'target_clash': [], 'rmsd': []}

        for file in os.listdir(dock_output_path):
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
                    in_place_data['name'].append(n)
                    in_place_data['glide_score'].append(results_by_ligand[n][0]['Score'])
                    score = score_no_vdW(results_by_ligand[n][0])
                    in_place_data['score_no_vdw'].append(score)
                    if score > 20:
                        in_place_data['modified_score_no_vdw'].append(20)
                    elif score < -20:
                        in_place_data['modified_score_no_vdw'].append(-20)
                    else:
                        in_place_data['modified_score_no_vdw'].append(score)
                    in_place_data['start_clash'].append(df[df['name'] == n]['start_clash'].iloc[0])
                    in_place_data['target_clash'].append(df[df['name'] == n]['target_clash'].iloc[0])
                    in_place_data['rmsd'].append(df[df['name'] == n]['rmsd'].iloc[0])

        in_place_df = pd.DataFrame.from_dict(in_place_data)
        in_place_df.to_csv(os.path.join(pose_path, 'in_place_combined.csv'))

    elif args.task == 'add_glide':
        pair = '{}-to-{}'.format(args.target, args.start)
        protein_path = os.path.join(args.raw_root, args.protein)
        pair_path = os.path.join(protein_path, pair)
        pose_path = os.path.join(pair_path, args.decoy_type)

        grid_df = pd.read_csv(os.path.join(pose_path, 'in_place_combined.csv'))
        glide_df = pd.read_csv(os.path.join(pair_path, '{}.csv'.format(pair)))
        glide_data = {'name': [], 'rmsd': [], 'glide_score': [], 'score_no_vdw': [], 'modified_score_no_vdw': []}

        for i in range(1, 100):
            pose_df = glide_df[glide_df['target'] == '{}_lig{}'.format(args.target, i)]
            if len(pose_df) > 0:
                glide_data['name'].append(pose_df['target'].iloc[0])
                glide_data['rmsd'].append(pose_df['rmsd'].iloc[0])
                glide_data['glide_score'].append(pose_df['glide_score'].iloc[0])
                score = pose_df['score_no_vdw'].iloc[0]
                glide_data['score_no_vdw'].append(score)
                if score > 20:
                    glide_data['modified_score_no_vdw'].append(20)
                elif score < -20:
                    glide_data['modified_score_no_vdw'].append(-20)
                else:
                    glide_data['modified_score_no_vdw'].append(score)

        df_to_add = pd.DataFrame.from_dict(glide_data)
        df = pd.concat([grid_df, df_to_add])
        df.to_csv(os.path.join(pose_path, 'combined_glide.csv'))

    elif args.task == 'stats':
        pair = '{}-to-{}'.format(args.target, args.start)
        protein_path = os.path.join(args.raw_root, args.protein)
        pair_path = os.path.join(protein_path, pair)
        pose_path = os.path.join(pair_path, args.decoy_type)

        df = pd.read_csv(os.path.join(pose_path, 'in_place_combined.csv'))
        print(len(df[df['rmsd'] < 2]))
        print(len(df))
        print(len(df[df['rmsd'] < 2]) / len(df))

if __name__=="__main__":
    main()