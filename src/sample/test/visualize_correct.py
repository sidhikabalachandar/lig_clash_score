"""
The purpose of this code is to first create the raw directory folder and include the following files
starting protein receptor
starting ligand
target ligand
glide pose viewer file

Then the top glide poses are added

Then the decoys are created

It can be run on sherlock using
$ $SCHRODINGER/run python3 visualize_correct.py /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw --group_name exhaustive_grid_4_2_rotation_0_360_20_rmsd_2.5
"""

import argparse
import os
import random
import pandas as pd
import math
import schrodinger.structure as structure
from schrodinger.structutils.transform import get_centroid
import sys
sys.path.insert(1, '../util')
from util import *
from schrod_replacement_util import *

X_AXIS = [1.0, 0.0, 0.0]  # x-axis unit vector
Y_AXIS = [0.0, 1.0, 0.0]  # y-axis unit vector
Z_AXIS = [0.0, 0.0, 1.0]  # z-axis unit vector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('--group_name', type=str, default='', help='name of pose group subdir')
    parser.add_argument('--residue_cutoff', type=int, default=0, help='name of pose group subdir')
    args = parser.parse_args()
    random.seed(0)

    protein = 'P00797'
    target = '3own'
    start = '3d91'
    pair = '{}-to-{}'.format(target, start)
    protein_path = os.path.join(args.raw_root, protein)
    pair_path = os.path.join(protein_path, pair)
    pose_path = os.path.join(pair_path, args.group_name)
    correct_path = os.path.join(pose_path, 'correct_after_simple_filter')
    clash_path = os.path.join(correct_path, 'clash_data')
    file = os.path.join(clash_path, 'combined.csv')
    start_df = pd.read_csv(file)
    correct_df = start_df[start_df['pred_num_intolerable'] <= args.residue_cutoff]
    incorrect_df = start_df[start_df['pred_num_intolerable'] > args.residue_cutoff]

    conformer_file = os.path.join(pair_path, "aligned_to_start_with_hydrogen_conformers.mae")
    conformers = list(structure.StructureReader(conformer_file))

    start_prot_file = os.path.join(pair_path, '{}_prot.mae'.format(start))
    start_prot = list(structure.StructureReader(start_prot_file))[0]

    start_prot_grid, start_origin = get_grid(start_prot)

    tolerable_indices = [i for i in correct_df.index]
    random.shuffle(tolerable_indices)
    tolerable_poses = tolerable_indices[:100]

    intolerable_indices = [i for i in incorrect_df.index]
    random.shuffle(intolerable_indices)
    intolerable_poses = intolerable_indices[:100]

    for clash_cutoff in range(1, 5):
        num_without_clash = 0

        with structure.StructureWriter(os.path.join(clash_path,
                                'correct_tolerable_clash_revised_clash_cutoff_{}.mae'.format(clash_cutoff))) as correct:
            for i in tolerable_poses:
                conformer_index = correct_df.loc[[i]]['conformer_index'].iloc[0]
                c = conformers[conformer_index]
                old_coords = c.getXYZ(copy=True)
                grid_loc_x = correct_df.loc[[i]]['grid_loc_x'].iloc[0]
                grid_loc_y = correct_df.loc[[i]]['grid_loc_y'].iloc[0]
                grid_loc_z = correct_df.loc[[i]]['grid_loc_z'].iloc[0]
                translate_structure(c, grid_loc_x, grid_loc_y, grid_loc_z)
                conformer_center = list(get_centroid(c))
                coords = c.getXYZ(copy=True)
                rot_x = correct_df.loc[[i]]['rot_x'].iloc[0]
                rot_y = correct_df.loc[[i]]['rot_y'].iloc[0]
                rot_z = correct_df.loc[[i]]['rot_z'].iloc[0]

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

                start_clash = get_clash(c, start_prot_grid, start_origin)
                if start_clash <= clash_cutoff:
                    correct.append(c)
                    num_without_clash += 1

                c.setXYZ(old_coords)

        print('tolerable: clash cutoff = {}, num without clash = {}, num total = {}'.format(clash_cutoff,
                                                                                            num_without_clash,
                                                                                            len(tolerable_poses)))
        num_without_clash = 0

        with structure.StructureWriter(os.path.join(clash_path,
                            'correct_intolerable_clash_revised_clash_cutoff_{}.mae'.format(clash_cutoff))) as incorrect:
            indices = [i for i in incorrect_df.index]
            random.shuffle(indices)
            indices = indices[:100]
            for i in intolerable_poses:
                conformer_index = incorrect_df.loc[[i]]['conformer_index'].iloc[0]
                c = conformers[conformer_index]
                old_coords = c.getXYZ(copy=True)
                grid_loc_x = incorrect_df.loc[[i]]['grid_loc_x'].iloc[0]
                grid_loc_y = incorrect_df.loc[[i]]['grid_loc_y'].iloc[0]
                grid_loc_z = incorrect_df.loc[[i]]['grid_loc_z'].iloc[0]
                translate_structure(c, grid_loc_x, grid_loc_y, grid_loc_z)
                conformer_center = list(get_centroid(c))
                coords = c.getXYZ(copy=True)
                rot_x = incorrect_df.loc[[i]]['rot_x'].iloc[0]
                rot_y = incorrect_df.loc[[i]]['rot_y'].iloc[0]
                rot_z = incorrect_df.loc[[i]]['rot_z'].iloc[0]

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

                start_clash = get_clash(c, start_prot_grid, start_origin)
                if start_clash <= clash_cutoff:
                    incorrect.append(c)
                    num_without_clash += 1

                c.setXYZ(old_coords)
            print('intolerable: clash cutoff = {}, num without clash = {}, num total = {}'.format(clash_cutoff,
                                                                            num_without_clash, len(intolerable_poses)))


if __name__ == "__main__":
    main()
