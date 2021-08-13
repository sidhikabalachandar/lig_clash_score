"""
The purpose of this code is to create the cumulative frequency and bar graphs

It can be run on sherlock using
$ $SCHRODINGER/run python3 python_score.py all /home/users/sidhikab/lig_clash_score/src/sample/test/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw --protein P02829 --target 2weq --start 2yge --index 0
"""

import argparse
import os
import pandas as pd
import schrodinger.structure as structure
import random
from schrodinger.structutils.transform import get_centroid
import math
import matplotlib.pyplot as plt
import sys
import numpy as np
sys.path.insert(1, '../util')
from util import *
from prot_util import *
from schrod_replacement_util import *
sys.path.insert(1, '../../../../physics_scoring')
from score_np import *


X_AXIS = [1.0, 0.0, 0.0]  # x-axis unit vector
Y_AXIS = [0.0, 1.0, 0.0]  # y-axis unit vector
Z_AXIS = [0.0, 0.0, 1.0]  # z-axis unit vector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either align or search')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('--protein', type=str, default='', help='protein name')
    parser.add_argument('--target', type=str, default='', help='target ligand name')
    parser.add_argument('--start', type=str, default='', help='start ligand name')
    parser.add_argument('--max_poses', type=int, default=100, help='maximum number of glide poses considered')
    parser.add_argument('--num_conformers', type=int, default=300, help='maximum number of conformers considered')
    parser.add_argument('--rotation_search_step_size', type=int, default=5, help='step size between each angle '
                                                                                 'checked, in degrees')
    parser.add_argument('--grid_size', type=int, default=1, help='grid size in positive and negative x, y, z '
                                                                 'directions')
    parser.add_argument('--target_clash_cutoff', type=int, default=20, help='clash cutoff between target protein and '
                                                                            'ligand pose')
    parser.add_argument('--intolerable_cutoff', type=int, default=0, help='cutoff of max num intolerable residues')
    parser.add_argument('--index', type=int, default=-1, help='index of pose file')
    parser.add_argument('--n', type=int, default=30, help='number of files processed in each job')
    args = parser.parse_args()

    random.seed(0)

    if args.task == 'all':
        counter = 0
        for protein, target, start in [('P00797', '3own', '3d91'), ('C8B467', '5ult', '5uov')]:
            pair = '{}-to-{}'.format(target, start)
            protein_path = os.path.join(args.raw_root, protein)
            pair_path = os.path.join(protein_path, pair)

            grid_size = get_grid_size(pair_path, target, start)
            group_name = 'test_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size)
            pose_path = os.path.join(pair_path, group_name)

            df = pd.read_csv(os.path.join(pose_path, 'poses_after_advanced_filter.csv'))
            names = df['name'].to_list()
            grouped_names = group_files(args.n, names)

            for i in range(len(grouped_names)):
                cmd = 'sbatch -p rondror -t 0:20:00 -o {} --wrap="$SCHRODINGER/run python3 python_score.py group {} {} ' \
                      '--protein {} --target {} --start {} --index {}"'
                counter += 1
                # os.system(cmd.format(os.path.join(args.run_path, 'score_{}.out'.format(i)), args.run_path, args.raw_root,
                #                      args.protein, args.target, args.start, i))

        print(counter)

    elif args.task == 'group':
        pair = '{}-to-{}'.format(args.target, args.start)
        protein_path = os.path.join(args.raw_root, args.protein)
        pair_path = os.path.join(protein_path, pair)

        grid_size = get_grid_size(pair_path, args.target, args.start)
        group_name = 'test_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size)
        pose_path = os.path.join(pair_path, group_name)

        conformer_file = os.path.join(pair_path, "aligned_to_start_with_hydrogen_conformers.mae")
        conformers = list(structure.StructureReader(conformer_file))

        df = pd.read_csv(os.path.join(pose_path, 'poses_after_advanced_filter.csv'))
        names = df['name'].to_list()
        grouped_names = group_files(args.n, names)
        group_names = grouped_names[args.index]
        scores = []

        protein_file = os.path.join(pair_path, '{}_prot.mae'.format(args.start))
        prot_s = list(structure.StructureReader(protein_file))[0]
        target_coord = prot_s.getXYZ(copy=True)
        target_charge = np.array([a.partial_charge for a in prot_s.atom])
        target_atom_type = [a.element for a in prot_s.atom]

        for name in group_names:
            conformer_index = df[df['name'] == name]['conformer_index'].iloc[0]
            c = conformers[conformer_index]
            old_coords = c.getXYZ(copy=True)
            grid_loc_x = df[df['name'] == name]['grid_loc_x'].iloc[0]
            grid_loc_y = df[df['name'] == name]['grid_loc_y'].iloc[0]
            grid_loc_z = df[df['name'] == name]['grid_loc_z'].iloc[0]
            translate_structure(c, grid_loc_x, grid_loc_y, grid_loc_z)
            conformer_center = list(get_centroid(c))
            coords = c.getXYZ(copy=True)
            rot_x = df[df['name'] == name]['rot_x'].iloc[0]
            rot_y = df[df['name'] == name]['rot_y'].iloc[0]
            rot_z = df[df['name'] == name]['rot_z'].iloc[0]

            displacement_vector = get_coords_array_from_list(conformer_center)
            to_origin_matrix = get_translation_matrix(-1 * displacement_vector)
            from_origin_matrix = get_translation_matrix(displacement_vector)
            rot_matrix_x = get_rotation_matrix(X_AXIS, math.radians(rot_x))
            rot_matrix_y = get_rotation_matrix(Y_AXIS, math.radians(rot_y))
            rot_matrix_z = get_rotation_matrix(Z_AXIS, math.radians(rot_z))
            new_coords = rotate_structure(coords, from_origin_matrix, to_origin_matrix, rot_matrix_x,
                                          rot_matrix_y, rot_matrix_z)

            # for clash features dictionary
            c.setXYZ(new_coords)
            c.title = name

            ligand_coord = new_coords
            ligand_charge = np.array([a.partial_charge for a in c.atom])
            ligand_atom_type = [a.element for a in c.atom]
            score = physics_score(ligand_coord, ligand_charge, target_coord, target_charge, ligand_atom_type,
                                  target_atom_type, vdw_scale=0)
            scores.append(score)

            c.setXYZ(old_coords)

        group_df = df.loc[df['name'].isin(group_names)]
        group_df['np_score_no_vdw'] = scores
        save_path = os.path.join(pose_path, 'poses_after_advanced_filter')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        group_df.to_csv(os.path.join(save_path, '{}.csv'.format(args.index)))

    elif args.task == 'combine':
        pair = '{}-to-{}'.format(args.target, args.start)
        protein_path = os.path.join(args.raw_root, args.protein)
        pair_path = os.path.join(protein_path, pair)

        grid_size = get_grid_size(pair_path, args.target, args.start)
        group_name = 'test_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size)
        pose_path = os.path.join(pair_path, group_name)

        combined_pose_file = os.path.join(pose_path, 'poses_after_advanced_filter.csv')
        df = pd.read_csv(combined_pose_file)
        names = df['name'].to_list()
        grouped_names = group_files(args.n, names)
        dfs = []

        for i in range(len(grouped_names)):
            save_path = os.path.join(pose_path, 'poses_after_advanced_filter')
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            file = os.path.join(save_path, '{}.csv'.format(i))
            dfs.append(pd.read_csv(file))

        df = pd.concat(dfs)
        df.to_csv(combined_pose_file)

    elif args.task == 'graph':
        pair = '{}-to-{}'.format(args.target, args.start)
        protein_path = os.path.join(args.raw_root, args.protein)
        pair_path = os.path.join(protein_path, pair)

        grid_size = get_grid_size(pair_path, args.target, args.start)
        group_name = 'test_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size)
        pose_path = os.path.join(pair_path, group_name)

        combined_pose_file = os.path.join(pose_path, 'poses_after_advanced_filter.csv')
        df = pd.read_csv(combined_pose_file)

        fig, ax = plt.subplots()
        plt.scatter(df['modified_score_no_vdw'].to_list(), df['np_score_no_vdw'].to_list())

        ax.legend()
        ax.set_xlabel('modified_score_no_vdw')
        ax.set_ylabel('np_score_no_vdw')
        plt.title('Glide score vs np score comparisson for {}_{}-to-{}'.format(args.protein, args.target, args.start))
        plt.savefig('score_{}_{}_{}.png'.format(args.protein, args.target, args.start))

    elif args.task == 'glide':
        pair = '{}-to-{}'.format(args.target, args.start)
        protein_path = os.path.join(args.raw_root, args.protein)
        pair_path = os.path.join(protein_path, pair)

        grid_size = get_grid_size(pair_path, args.target, args.start)
        group_name = 'test_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size)
        pose_path = os.path.join(pair_path, group_name)

        glide_df = pd.read_csv(os.path.join(pair_path, '{}.csv'.format(pair)))
        names = []
        scores = []

        protein_file = os.path.join(pair_path, '{}_prot.mae'.format(args.start))
        prot_s = list(structure.StructureReader(protein_file))[0]
        target_coord = prot_s.getXYZ(copy=True)
        target_charge = np.array([a.partial_charge for a in prot_s.atom])
        target_atom_type = [a.element for a in prot_s.atom]

        for i in range(1, 100):
            name = '{}_lig{}'.format(args.target, i)
            pose_file = os.path.join(pair_path, 'ligand_poses', '{}.mae'.format(name))
            c = list(structure.StructureReader(pose_file))[0]
            ligand_coord = c.getXYZ(copy=True)
            ligand_charge = np.array([a.partial_charge for a in c.atom])
            ligand_atom_type = [a.element for a in c.atom]
            score = physics_score(ligand_coord, ligand_charge, target_coord, target_charge, ligand_atom_type,
                                  target_atom_type, vdw_scale=0)
            names.append(name)
            scores.append(score)

        glide_df = glide_df.loc[glide_df['target'].isin(names)]
        glide_df['np_score_no_vdw'] = scores
        glide_df.to_csv(os.path.join(pose_path, 'glide_poses.csv'))


if __name__=="__main__":
    main()