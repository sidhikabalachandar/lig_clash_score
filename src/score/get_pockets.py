"""
The purpose of this code is to get the physics scores and the rmsds

It can be run on sherlock using
$ $SCHRODINGER/run python3 get_pockets.py run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits/search_test_incorrect_glide_index.txt /home/users/sidhikab/lig_clash_score/src/sample/test/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw --max_num_concurrent_jobs 1
"""

import argparse
import os
import schrodinger.structure as structure
import pandas as pd

import sys
sys.path.append('/home/users/sidhikab/docking')
from docking.docking_class import Docking_Set
sys.path.insert(1, '../util')
from util import *
from prot_util import *
from schrod_replacement_util import *
from lig_util import *


X_AXIS = [1.0, 0.0, 0.0]  # x-axis unit vector
Y_AXIS = [0.0, 1.0, 0.0]  # y-axis unit vector
Z_AXIS = [0.0, 0.0, 1.0]  # z-axis unit vector


def run(protein, target, start, args):
    """
    get scores and rmsds
    """

    pair = '{}-to-{}'.format(target, start)
    protein_path = os.path.join(args.raw_root, protein)
    pair_path = os.path.join(protein_path, pair)

    grid_size = get_grid_size(pair_path, target, start)
    group_name = 'test_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size)
    pose_path = os.path.join(pair_path, group_name)

    clash_path = os.path.join(pose_path, 'clash_data')
    dfs = []
    for file in os.listdir(clash_path):
        prefix = 'pose_pred_data'
        if file[:len(prefix)] == prefix:
            df = pd.read_csv(os.path.join(clash_path, file))
            filter_df = df[df['pred_num_intolerable'] < args.residue_cutoff]
            dfs.append(filter_df)

    df = pd.concat(dfs)
    names = df['name'].to_list()
    grouped_names = group_files(args.n, names)

    conformer_file = os.path.join(pair_path, "aligned_to_start_with_hydrogen_conformers.mae")
    conformers = list(structure.StructureReader(conformer_file))

    grouped_path = os.path.join(pose_path, 'advanced_filtered_poses')
    dock_output_path = os.path.join(pose_path, 'dock_output')
    ground_truth_file = os.path.join(pair_path, 'ligand_poses', '{}_lig0.mae'.format(target))

    if not os.path.exists(grouped_path):
        os.mkdir(grouped_path)

    if not os.path.exists(dock_output_path):
        os.mkdir(dock_output_path)

    docking_config = []

    for j in range(len(grouped_names)):
        file = os.path.join(grouped_path, '{}.mae'.format(j))
        with structure.StructureWriter(file) as filtered:
            for name in grouped_names[j]:
                conformer_index = df[df['name'] == name]['conformer_index'].iloc[0]
                c = conformers[conformer_index]
                old_coords = c.getXYZ(copy=True)
                grid_loc_x = df[df['name'] == name]['grid_loc_x'].iloc[0]
                grid_loc_y = df[df['name'] == name]['grid_loc_y'].iloc[0]
                grid_loc_z = df[df['name'] == name]['grid_loc_z'].iloc[0]
                rot_x = df[df['name'] == name]['rot_x'].iloc[0]
                rot_y = df[df['name'] == name]['rot_y'].iloc[0]
                rot_z = df[df['name'] == name]['rot_z'].iloc[0]

                new_coords = create_pose(c, grid_loc_x, grid_loc_y, grid_loc_z, rot_x, rot_y, rot_z)

                # for clash features dictionary
                c.setXYZ(new_coords)
                c.title = name
                filtered.append(c)
                c.setXYZ(old_coords)

        if not os.path.exists(os.path.join(dock_output_path, '{}.scor'.format(j))):
            docking_config.append({'folder': dock_output_path,
                                   'name': j,
                                   'grid_file': os.path.join(pair_path, '{}.zip'.format(pair)),
                                   'prepped_ligand_file': file,
                                   'glide_settings': {'num_poses': 1, 'docking_method': 'inplace'},
                                   'ligand_file': ground_truth_file})
        if len(docking_config) == args.max_num_concurrent_jobs:
            break

    print(len(docking_config))

    run_config = {'run_folder': args.run_path,
                  'group_size': 1,
                  'partition': 'rondror',
                  'dry_run': False}

    dock_set = Docking_Set()
    dock_set.run_docking_rmsd_delete(docking_config, run_config)


def check(protein, target, start, args):
    """
    check if scores and rmsds were calculated
    :param docked_prot_file: (string) file listing proteins to process
    :param raw_root: (string) directory where raw data will be placed
    :return:
    """

    pair = '{}-to-{}'.format(target, start)
    protein_path = os.path.join(args.raw_root, protein)
    pair_path = os.path.join(protein_path, pair)

    grid_size = get_grid_size(pair_path, target, start)
    group_name = 'test_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size)
    pose_path = os.path.join(pair_path, group_name)

    grouped_path = os.path.join(pose_path, 'advanced_filtered_poses')
    dock_output_path = os.path.join(pose_path, 'dock_output')
    ground_truth_file = os.path.join(pair_path, 'ligand_poses', '{}_lig0.mae'.format(target))

    missing = []
    incomplete = []

    for file_name in os.listdir(grouped_path):
        suffix = '.mae'
        name = file_name[:-len(suffix)]
        file = os.path.join(grouped_path, file_name)
        docking_config = [{'folder': dock_output_path,
                           'name': name,
                           'grid_file': os.path.join(pair_path, '{}.zip'.format(pair)),
                           'prepped_ligand_file': file,
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
            group = list(structure.StructureReader(file))
            if len(results_by_ligand.keys()) != len(group):
                print(len(results_by_ligand), len(group))
                incomplete.append(name)
                continue

    print('Missing', len(missing), '/', len(os.listdir(grouped_path)))
    print('Incomplete', len(incomplete), '/', len(os.listdir(grouped_path)) - len(missing))
    print(incomplete)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either run, check, or delete_json')
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('--protein', type=str, default='', help='protein name')
    parser.add_argument('--target', type=str, default='', help='target ligand name')
    parser.add_argument('--start', type=str, default='', help='start ligand name')
    parser.add_argument('--n', type=int, default=35, help='number of protein, target, start groups processed in '
                                                         'group task')
    parser.add_argument('--max_num_concurrent_jobs', type=int, default=200, help='maximum number of concurrent jobs '
                                                                                 'that can be run on slurm at one time')
    parser.add_argument('--residue_cutoff', type=int, default=3, help='name of pose group subdir')
    args = parser.parse_args()

    if not os.path.exists(args.run_path):
        os.mkdir(args.run_path)

    if args.task == 'run':
        pairs = get_prots(args.docked_prot_file)
        for protein, target, start in pairs[:10]:
            pair = '{}-to-{}'.format(target, start)
            protein_path = os.path.join(args.raw_root, protein)
            pair_path = os.path.join(protein_path, pair)

            grid_size = get_grid_size(pair_path, target, start)
            group_name = 'test_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size)
            pose_path = os.path.join(pair_path, group_name)
            file = os.path.join(pose_path, 'poses_after_advanced_filter.csv')
            df = pd.read_csv(file)
            print(protein, target, start, len(df))
        #
        # clash_path = os.path.join(pose_path, 'clash_data')
        # dfs = []
        # for file in os.listdir(clash_path):
        #     prefix = 'pose_pred_data'
        #     if file[:len(prefix)] == prefix:
        #         df = pd.read_csv(os.path.join(clash_path, file))
        #         filter_df = df[df['pred_num_intolerable'] < args.residue_cutoff]
        #         dfs.append(filter_df)
        #
        # df = pd.concat(dfs)
        # names = df['name'].to_list()
        # grouped_names = group_files(args.n, names)
        #
        # conformer_file = os.path.join(pair_path, "aligned_to_start_with_hydrogen_conformers.mae")
        # conformers = list(structure.StructureReader(conformer_file))
        #
        # grouped_path = os.path.join(pose_path, 'advanced_filtered_poses')
        # dock_output_path = os.path.join(pose_path, 'dock_output')
        # ground_truth_file = os.path.join(pair_path, 'ligand_poses', '{}_lig0.mae'.format(target))
        #
        # if not os.path.exists(grouped_path):
        #     os.mkdir(grouped_path)
        #
        # if not os.path.exists(dock_output_path):
        #     os.mkdir(dock_output_path)
        #
        # docking_config = []
        #
        # for j in range(len(grouped_names)):
        #     file = os.path.join(grouped_path, '{}.mae'.format(j))
        #     with structure.StructureWriter(file) as filtered:
        #         for name in grouped_names[j]:
        #             conformer_index = df[df['name'] == name]['conformer_index'].iloc[0]
        #             c = conformers[conformer_index]
        #             old_coords = c.getXYZ(copy=True)
        #             grid_loc_x = df[df['name'] == name]['grid_loc_x'].iloc[0]
        #             grid_loc_y = df[df['name'] == name]['grid_loc_y'].iloc[0]
        #             grid_loc_z = df[df['name'] == name]['grid_loc_z'].iloc[0]
        #             rot_x = df[df['name'] == name]['rot_x'].iloc[0]
        #             rot_y = df[df['name'] == name]['rot_y'].iloc[0]
        #             rot_z = df[df['name'] == name]['rot_z'].iloc[0]
        #
        #             new_coords = create_pose(c, grid_loc_x, grid_loc_y, grid_loc_z, rot_x, rot_y, rot_z)
        #
        #             # for clash features dictionary
        #             c.setXYZ(new_coords)
        #             c.title = name
        #             filtered.append(c)
        #             c.setXYZ(old_coords)
        #
        #     if not os.path.exists(os.path.join(dock_output_path, '{}.scor'.format(j))):
        #         docking_config.append({'folder': dock_output_path,
        #                                'name': j,
        #                                'grid_file': os.path.join(pair_path, '{}.zip'.format(pair)),
        #                                'prepped_ligand_file': file,
        #                                'glide_settings': {'num_poses': 1, 'docking_method': 'inplace'},
        #                                'ligand_file': ground_truth_file})
        #     if len(docking_config) == args.max_num_concurrent_jobs:
        #         break
        #
        # print(len(docking_config))
        #
        # run_config = {'run_folder': args.run_path,
        #               'group_size': 1,
        #               'partition': 'rondror',
        #               'dry_run': False}
        #
        # dock_set = Docking_Set()
        # dock_set.run_docking_rmsd_delete(docking_config, run_config)

    elif args.task == 'check':
        for protein, target, start in [('P02829', '2weq', '2yge'), ('P00797', '3own', '3d91'),
                                       ('C8B467', '5ult', '5uov')]:
            check(protein, target, start, args)

    elif args.task == 'remove':
        for protein, target, start in [('P02829', '2weq', '2yge'), ('P00797', '3own', '3d91'),
                                       ('C8B467', '5ult', '5uov')]:
            pair = '{}-to-{}'.format(target, start)
            protein_path = os.path.join(args.raw_root, protein)
            pair_path = os.path.join(protein_path, pair)

            grid_size = get_grid_size(pair_path, target, start)
            group_name = 'test_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size)
            pose_path = os.path.join(pair_path, group_name)
            save_path = os.path.join(pose_path, 'dock_output')
            os.system('rm -rf {}'.format(save_path))
            save_path = os.path.join(pose_path, 'advanced_filtered_poses')
            os.system('rm -rf {}'.format(save_path))

if __name__=="__main__":
    main()