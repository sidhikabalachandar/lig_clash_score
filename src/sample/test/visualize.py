

"""
The purpose of this code is to create conformers

It can be run on sherlock using
$ $SCHRODINGER/run python3 clash_search.py /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits/search_test_incorrect_glide_index.txt /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
"""

import argparse
import os
import random
import pandas as pd
import schrodinger.structure as structure
from schrodinger.structutils.transform import get_centroid
import math
import sys
sys.path.insert(1, '../util')
from util import *
from prot_util import *


X_AXIS = [1.0, 0.0, 0.0]  # x-axis unit vector
Y_AXIS = [0.0, 1.0, 0.0]  # y-axis unit vector
Z_AXIS = [0.0, 0.0, 1.0]  # z-axis unit vector


def write_poses(protein, pair, df, conformers, name):
    with structure.StructureWriter('{}_{}_{}.mae'.format(protein, pair, name)) as correct:
        indices = [i for i in df.index]
        random.shuffle(indices)
        for i in indices[:100]:
            conformer_index = df.loc[[i]]['conformer_index'].iloc[0]
            c = conformers[conformer_index]
            old_coords = c.getXYZ(copy=True)
            grid_loc_x = df.loc[[i]]['grid_loc_x'].iloc[0]
            grid_loc_y = df.loc[[i]]['grid_loc_y'].iloc[0]
            grid_loc_z = df.loc[[i]]['grid_loc_z'].iloc[0]
            translate_structure(c, grid_loc_x, grid_loc_y, grid_loc_z)
            conformer_center = list(get_centroid(c))
            coords = c.getXYZ(copy=True)
            rot_x = df.loc[[i]]['rot_x'].iloc[0]
            rot_y = df.loc[[i]]['rot_y'].iloc[0]
            rot_z = df.loc[[i]]['rot_z'].iloc[0]

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
            correct.append(c)
            c.setXYZ(old_coords)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_root', type=str, help='directory where script and output files will be written')
    parser.add_argument('--residue_cutoff', type=int, default=2, help='name of pose group subdir')
    parser.add_argument('--rmsd_cutoff', type=float, default=2.5, help='name of pose group subdir')
    parser.add_argument('--start_clash_cutoff', type=int, default=1, help='clash cutoff between start protein and '
                                                                          'ligand pose')
    args = parser.parse_args()
    random.seed(0)

    protein = 'P00797'
    target = '3own'
    start = '3d91'
    pair = '{}-to-{}'.format(target, start)
    protein_path = os.path.join(args.raw_root, protein)
    pair_path = os.path.join(protein_path, pair)

    grid_size = get_grid_size(pair_path, target, start)
    group_name = 'test_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size)
    pose_path = os.path.join(pair_path, group_name)
    clash_path = os.path.join(pose_path, 'clash_data')

    conformer_file = os.path.join(pair_path, "aligned_to_start_with_hydrogen_conformers.mae")
    conformers = list(structure.StructureReader(conformer_file))

    dfs = []
    for file in os.listdir(clash_path):
        prefix = 'pose_pred_data'
        if file[:len(prefix)] == prefix:
            df = pd.read_csv(os.path.join(clash_path, file))
            dfs.append(df)

    all_df = pd.concat(dfs)
    tolerable_df = all_df[all_df['pred_num_intolerable'] < args.residue_cutoff]
    tolerable_correct_df = tolerable_df[tolerable_df['rmsd'] < args.rmsd_cutoff]
    tolerable_incorrect_df = tolerable_df[tolerable_df['rmsd'] >= args.rmsd_cutoff]
    intolerable_df = all_df[all_df['pred_num_intolerable'] >= args.residue_cutoff]
    intolerable_correct_df = intolerable_df[intolerable_df['rmsd'] < args.rmsd_cutoff]
    intolerable_incorrect_df = intolerable_df[intolerable_df['rmsd'] >= args.rmsd_cutoff]

    write_poses(protein, pair, tolerable_correct_df, conformers, 'tolerable_correct')
    write_poses(protein, pair, tolerable_incorrect_df, conformers, 'tolerable_incorrect')
    write_poses(protein, pair, intolerable_correct_df, conformers, 'intolerable_correct')
    write_poses(protein, pair, intolerable_incorrect_df, conformers, 'intolerable_incorrect')
    

if __name__ == "__main__":
    main()

