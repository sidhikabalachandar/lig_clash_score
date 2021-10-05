"""
The purpose of this code is to get the physics scores and the rmsds

It can be run on sherlock using
$ $SCHRODINGER/run python3 get_ligands.py balanced /home/users/sidhikab/lig_clash_score/src/score/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d --data_name balanced_data --protein P02829 --target 2weq --start 2yge
"""

import argparse
import os
import schrodinger.structure as structure
import pandas as pd
import random

import sys
sys.path.insert(1, '../sample/util')
from util import *
from prot_util import *
from schrod_replacement_util import *
from lig_util import *
from prepare_pockets import *

X_AXIS = [1.0, 0.0, 0.0]  # x-axis unit vector
Y_AXIS = [0.0, 1.0, 0.0]  # y-axis unit vector
Z_AXIS = [0.0, 0.0, 1.0]  # z-axis unit vector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either run, check, or delete_json')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('root', type=str, help='directory where raw data will be placed')
    parser.add_argument('--protein', type=str, default='', help='protein name')
    parser.add_argument('--target', type=str, default='', help='target ligand name')
    parser.add_argument('--start', type=str, default='', help='start ligand name')
    parser.add_argument('--n', type=int, default=35, help='number of protein, target, start groups processed in '
                                                         'group task')
    parser.add_argument('--max_num_concurrent_jobs', type=int, default=200, help='maximum number of concurrent jobs '
                                                                                 'that can be run on slurm at one time')
    parser.add_argument('--residue_cutoff', type=int, default=3, help='name of pose group subdir')
    parser.add_argument('--data_name', type=str, default='data', help='name of saved data folder')
    parser.add_argument('--rmsd_cutoff', type=float, default=2.5, help='name of pose group subdir')
    args = parser.parse_args()

    random.seed(0)

    raw_root = os.path.join(args.root, 'raw')
    save_directory = os.path.join(args.root, 'ml_score')
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)
    save_directory = os.path.join(save_directory, args.data_name)
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    if not os.path.exists(args.run_path):
        os.mkdir(args.run_path)

    if args.task == 'all':
        for protein, target, start in [('P02829', '2weq', '2yge'), ('P00797', '3own', '3d91'),
                                       ('C8B467', '5ult', '5uov'),
                                       ('P03368', '1gno', '1zp8'), ('P02829', '2fxs', '2weq'),
                                       ('P11838', '3wz6', '1gvx'),
                                       ('P00523', '4ybk', '2oiq'), ('P00519', '4twp', '5hu9'),
                                       ('P0DOX7', '6msy', '6mub')]:
            cmd = 'sbatch -p rondror -t 1:00:00 -o {} --wrap="$SCHRODINGER/run python3 get_ligands.py group {} {} ' \
                  '--protein {} --target {} --start {}"'
            os.system(
                cmd.format(os.path.join(args.run_path, 'ligands_{}_{}_{}.out'.format(protein, target, start)),
                           args.run_path, args.root, protein, target, start))

    if args.task == 'group':
        pair = '{}-to-{}'.format(args.target, args.start)
        protein_path = os.path.join(raw_root, args.protein)
        pair_path = os.path.join(protein_path, pair)

        grid_size = get_grid_size(pair_path, args.target, args.start)
        group_name = 'test_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size)
        pose_path = os.path.join(pair_path, group_name)
        file = os.path.join(pose_path, 'poses_after_advanced_filter.csv')
        df = pd.read_csv(file)

        names = df['name'].to_list()

        conformer_file = os.path.join(pair_path, "aligned_to_start_with_hydrogen_conformers.mae")
        conformers = list(structure.StructureReader(conformer_file))

        ligand_file = os.path.join(save_directory, '{}_{}_ligs.mae'.format(args.protein, pair))
        with structure.StructureWriter(ligand_file) as filtered:
            for name in names:
                pose_name = '{}_{}_{}'.format(args.protein, pair, name)
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
                c.title = pose_name
                filtered.append(c)
                c.setXYZ(old_coords)

    if args.task == 'small_data':
        for protein, target, start in [('P02829', '2weq', '2yge'), ('P00797', '3own', '3d91'),
                                       ('C8B467', '5ult', '5uov'),
                                       ('P03368', '1gno', '1zp8'), ('P02829', '2fxs', '2weq'),
                                       ('P11838', '3wz6', '1gvx'),
                                       ('P00523', '4ybk', '2oiq'), ('P00519', '4twp', '5hu9'),
                                       ('P0DOX7', '6msy', '6mub')]:
            pair = '{}-to-{}'.format(target, start)
            protein_path = os.path.join(raw_root, protein)
            pair_path = os.path.join(protein_path, pair)

            grid_size = get_grid_size(pair_path, target, start)
            group_name = 'test_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size)
            pose_path = os.path.join(pair_path, group_name)
            file = os.path.join(pose_path, 'poses_after_advanced_filter.csv')
            df = pd.read_csv(file)

            names = df['name'].to_list()

            conformer_file = os.path.join(pair_path, "aligned_to_start_with_hydrogen_conformers.mae")
            conformers = list(structure.StructureReader(conformer_file))

            ligand_file = os.path.join(save_directory, '{}_{}_ligs.mae'.format(protein, pair))
            with structure.StructureWriter(ligand_file) as filtered:
                for name in names[:3]:
                    pose_name = '{}_{}_{}'.format(protein, pair, name)
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
                    c.title = pose_name
                    filtered.append(c)
                    c.setXYZ(old_coords)

    if args.task == 'balanced':
        for protein, target, start in [('P02829', '2weq', '2yge'), ('P00797', '3own', '3d91'),
                                       ('C8B467', '5ult', '5uov'),
                                       ('P03368', '1gno', '1zp8'), ('P02829', '2fxs', '2weq'),
                                       ('P11838', '3wz6', '1gvx'),
                                       ('P00523', '4ybk', '2oiq'), ('P00519', '4twp', '5hu9'),
                                       ('P0DOX7', '6msy', '6mub')]:
            pair = '{}-to-{}'.format(target, start)
            protein_path = os.path.join(raw_root, protein)
            pair_path = os.path.join(protein_path, pair)

            grid_size = get_grid_size(pair_path, target, start)
            group_name = 'test_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size)
            pose_path = os.path.join(pair_path, group_name)
            file = os.path.join(pose_path, 'poses_after_advanced_filter.csv')
            df = pd.read_csv(file)
            correct_df = df[df['rmsd'] <= args.rmsd_cutoff]
            incorrect_df = df[df['rmsd'] > args.rmsd_cutoff]

            correct_names = correct_df['name'].to_list()
            incorrect_names = incorrect_df['name'].to_list()
            random.shuffle(incorrect_names)
            incorrect_names = incorrect_names[:len(correct_names)]
            names = correct_names + incorrect_names

            conformer_file = os.path.join(pair_path, "aligned_to_start_with_hydrogen_conformers.mae")
            conformers = list(structure.StructureReader(conformer_file))

            ligand_file = os.path.join(save_directory, '{}_{}_ligs.mae'.format(protein, pair))
            with structure.StructureWriter(ligand_file) as filtered:
                for name in names:
                    pose_name = '{}_{}_{}'.format(protein, pair, name)
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
                    c.title = pose_name
                    filtered.append(c)
                    c.setXYZ(old_coords)


if __name__=="__main__":
    main()