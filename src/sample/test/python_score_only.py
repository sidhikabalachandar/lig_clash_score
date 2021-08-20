"""
The purpose of this code is to create the cumulative frequency and bar graphs

It can be run on sherlock using
$ $SCHRODINGER/run python3 python_score_only.py group /home/users/sidhikab/lig_clash_score/src/sample/test/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/vdw_AMBER_parm99.defn --protein P11838 --target 3wz6 --start 1gvx --index 7 --n 5
"""

import argparse
import os
import pandas as pd
import schrodinger.structure as structure
import random
import numpy as np
import matplotlib.pyplot as plt
import schrodinger.structutils.interactions.steric_clash as steric_clash

import sys
sys.path.insert(1, '../util')
from util import *
from prot_util import *
from schrod_replacement_util import *
from lig_util import *
sys.path.insert(1, '../../../../physics_scoring')
from score_np import *
from read_vdw_params import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either align or search')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('vdw_param_file', type=str, help='directory where raw data will be placed')
    parser.add_argument('--protein', type=str, default='', help='protein name')
    parser.add_argument('--target', type=str, default='', help='target ligand name')
    parser.add_argument('--start', type=str, default='', help='start ligand name')
    parser.add_argument('--max_num_poses_considered', type=int, default=4000, help='maximum number of poses considered')
    parser.add_argument('--num_conformers', type=int, default=300, help='maximum number of conformers considered')
    parser.add_argument('--rotation_search_step_size', type=int, default=5, help='step size between each angle '
                                                                                 'checked, in degrees')
    parser.add_argument('--grid_size', type=int, default=1, help='grid size in positive and negative x, y, z '
                                                                 'directions')
    parser.add_argument('--target_clash_cutoff', type=int, default=20, help='clash cutoff between target protein and '
                                                                            'ligand pose')
    parser.add_argument('--intolerable_cutoff', type=int, default=0, help='cutoff of max num intolerable residues')
    parser.add_argument('--index', type=int, default=-1, help='index of pose file')
    parser.add_argument('--n', type=int, default=90, help='number of files processed in each job')
    parser.add_argument('--residue_cutoff', type=int, default=3, help='name of pose group subdir')
    parser.add_argument('--rmsd_cutoff', type=float, default=2.5, help='name of pose group subdir')
    args = parser.parse_args()

    random.seed(0)

    if not os.path.exists(args.run_path):
        os.mkdir(args.run_path)

    if args.task == 'all':
        counter = 0
        # for protein, target, start in [('P03368', '1gno', '1zp8'), ('P02829', '2fxs', '2weq'),
        #                                ('P11838', '3wz6', '1gvx'), ('P00523', '4ybk', '2oiq'),
        #                                ('P00519', '4twp', '5hu9'), ('P0DOX7', '6msy', '6mub')]:

        for protein, target, start, i in [('P11838', '3wz6', '1gvx', 7), ('P00519', '4twp', '5hu9', 1), ('P0DOX7', '6msy', '6mub', 16)]:
            # pair = '{}-to-{}'.format(target, start)
            # protein_path = os.path.join(args.raw_root, protein)
            # pair_path = os.path.join(protein_path, pair)

            # grid_size = get_grid_size(pair_path, target, start)
            # group_name = 'test_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size)
            # pose_path = os.path.join(pair_path, group_name)

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
            # correct_df = df[df['rmsd'] < args.rmsd_cutoff]
            # correct_names = correct_df['name'].to_list()
            # incorrect_df = df[df['rmsd'] <= args.rmsd_cutoff]
            # incorrect_names = incorrect_df['name'].to_list()
            # random.shuffle(incorrect_names)
            # incorrect_names = incorrect_names[:args.max_num_poses_considered - len(correct_names)]
            # names = correct_names + incorrect_names
            # grouped_names = group_files(args.n, names)

            # for i in range(len(grouped_names)):
            cmd = 'sbatch -p rondror -t 0:30:00 -o {} --wrap="$SCHRODINGER/run python3 python_score_only.py ' \
                  'group {} {} {} --protein {} --target {} --start {} --index {}"'
            counter += 1
            os.system(cmd.format(os.path.join(args.run_path,
                                              'score_{}_{}_{}_{}.out'.format(protein, target, start, i)),
                                 args.run_path, args.raw_root, args.vdw_param_file, protein, target, start, i))

        print(counter)

    elif args.task == 'group':
        pair = '{}-to-{}'.format(args.target, args.start)
        protein_path = os.path.join(args.raw_root, args.protein)
        pair_path = os.path.join(protein_path, pair)

        grid_size = get_grid_size(pair_path, args.target, args.start)
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
        correct_df = df[df['rmsd'] < args.rmsd_cutoff]
        correct_names = correct_df['name'].to_list()
        incorrect_df = df[df['rmsd'] <= args.rmsd_cutoff]
        incorrect_names = incorrect_df['name'].to_list()
        random.shuffle(incorrect_names)
        incorrect_names = incorrect_names[:args.max_num_poses_considered - len(correct_names)]
        names = correct_names + incorrect_names
        grouped_names = group_files(args.n, names)

        conformer_file = os.path.join(pair_path, "aligned_to_start_with_hydrogen_conformers.mae")
        conformers = list(structure.StructureReader(conformer_file))

        protein_file = os.path.join(pair_path, '{}_prot.mae'.format(args.start))
        prot_s = list(structure.StructureReader(protein_file))[0]
        target_coord = prot_s.getXYZ(copy=True)
        target_charge = np.array([a.partial_charge for a in prot_s.atom])
        target_atom_type = [a.element for a in prot_s.atom]
        vdw_params = read_vdw_params(args.vdw_param_file)

        group_df = df[df['name'].isin(grouped_names[args.index])]
        print(len(group_df))
        print(len(grouped_names[args.index]))
        print(len(group_df['name'].to_list()))

        counter = 0
        for name in grouped_names[args.index]:
            counter += 1
            if name not in group_df['name'].to_list():
                print(name)
                if name in correct_names:
                    print('correct')
                elif name in incorrect_names:
                    print('incorrect')

        print(counter)

        return

        python_scores = []

        for name in grouped_names[args.index]:
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

            no_clash_atom_indices = []
            for i in c.getAtomIndices():
                clash = steric_clash.clash_volume(prot_s, struc2=c, atoms2=[i])
                if clash == 0:
                    no_clash_atom_indices.append(i)

            no_clash_c = c.extract(no_clash_atom_indices)
            no_clash_ligand_coord = no_clash_c.getXYZ(copy=True)
            no_clash_ligand_charge = np.array([a.partial_charge for a in no_clash_c.atom])
            no_clash_ligand_atom_type = [a.element for a in no_clash_c.atom]
            python_score = physics_score(no_clash_ligand_coord, no_clash_ligand_charge, target_coord, target_charge,
                                         np.array(no_clash_ligand_atom_type), np.array(target_atom_type),
                                         vdw_params=vdw_params)

            c.setXYZ(old_coords)

            python_scores.append(python_score)

        group_df['python_score'] = python_scores

        save_path = os.path.join(pose_path, 'poses_after_advanced_filter')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        group_df.to_csv(os.path.join(save_path, '{}.csv'.format(args.index)))

    elif args.task == 'check':
        missing = []
        counter = 0

        for protein, target, start in [('P03368', '1gno', '1zp8'), ('P02829', '2fxs', '2weq'),
                                       ('P11838', '3wz6', '1gvx'), ('P00523', '4ybk', '2oiq'),
                                       ('P00519', '4twp', '5hu9'), ('P0DOX7', '6msy', '6mub')]:
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
            correct_df = df[df['rmsd'] < args.rmsd_cutoff]
            correct_names = correct_df['name'].to_list()
            incorrect_df = df[df['rmsd'] <= args.rmsd_cutoff]
            incorrect_names = incorrect_df['name'].to_list()
            random.shuffle(incorrect_names)
            incorrect_names = incorrect_names[:args.max_num_poses_considered - len(correct_names)]
            names = correct_names + incorrect_names
            grouped_names = group_files(args.n, names)

            for i in range(len(grouped_names)):
                counter += 1
                save_path = os.path.join(pose_path, 'poses_after_advanced_filter')
                file = os.path.join(save_path, '{}.csv'.format(i))
                if not os.path.exists(file):
                    missing.append((protein, target, start, i))

        print('Missing: {}/{}'.format(len(missing), counter))
        print(missing)

    elif args.task == 'combine':
        for protein, target, start in [('P03368', '1gno', '1zp8'), ('P02829', '2fxs', '2weq'),
                                       ('P11838', '3wz6', '1gvx'), ('P00523', '4ybk', '2oiq'),
                                       ('P00519', '4twp', '5hu9'), ('P0DOX7', '6msy', '6mub')]:
            pair = '{}-to-{}'.format(target, start)
            protein_path = os.path.join(args.raw_root, protein)
            pair_path = os.path.join(protein_path, pair)

            grid_size = get_grid_size(pair_path, target, start)
            group_name = 'test_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size)
            pose_path = os.path.join(pair_path, group_name)
            save_path = os.path.join(pose_path, 'poses_after_advanced_filter')

            dfs = []
            for f in os.listdir(save_path):
                file = os.path.join(save_path, f)
                dfs.append(pd.read_csv(file))

            combined_pose_file = os.path.join(pose_path, 'poses_after_advanced_filter.csv')
            df = pd.concat(dfs)
            df.to_csv(combined_pose_file)

    elif args.task == 'remove':
        for protein, target, start in [('P02829', '2weq', '2yge'), ('P00797', '3own', '3d91'),
                                       ('C8B467', '5ult', '5uov')]:
            pair = '{}-to-{}'.format(target, start)
            protein_path = os.path.join(args.raw_root, protein)
            pair_path = os.path.join(protein_path, pair)

            grid_size = get_grid_size(pair_path, target, start)
            group_name = 'test_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size)
            pose_path = os.path.join(pair_path, group_name)
            save_path = os.path.join(pose_path, 'poses_after_advanced_filter')
            os.system('rm -rf {}'.format(save_path))
            combined_pose_file = os.path.join(pose_path, 'poses_after_advanced_filter.csv')
            os.system('rm -rf {}'.format(combined_pose_file))

    elif args.task == 'graph':
        glide_scores = []
        python_scores = []
        for protein, target, start in [('P02829', '2weq', '2yge'), ('P00797', '3own', '3d91'),
                                       ('C8B467', '5ult', '5uov')]:
            pair = '{}-to-{}'.format(target, start)
            protein_path = os.path.join(args.raw_root, protein)
            pair_path = os.path.join(protein_path, pair)

            grid_size = get_grid_size(pair_path, target, start)
            group_name = 'test_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size)
            pose_path = os.path.join(pair_path, group_name)
            combined_pose_file = os.path.join(pose_path, 'poses_after_advanced_filter.csv')
            df = pd.read_csv(combined_pose_file)
            glide_scores.extend(df['modified_score_no_vdw'].to_list())
            python_scores.extend(df['python_score'].to_list())

        fig, ax = plt.subplots()
        plt.scatter(glide_scores, python_scores)
        plt.title('Glide score vs python score')
        plt.xlabel('Modified Glide Score No VDW')
        plt.ylabel('Python Score for Non-Clashing Ligand Atoms')
        ax.legend()
        fig.savefig('glide_vs_python.png')


if __name__=="__main__":
    main()