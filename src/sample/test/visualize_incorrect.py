"""
The purpose of this code is to first create the raw directory folder and include the following files
starting protein receptor
starting ligand
target ligand
glide pose viewer file

Then the top glide poses are added

Then the decoys are created

It can be run on sherlock using
$ $SCHRODINGER/run python3 visualize_incorrect.py /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw --group_name exhaustive_grid_4_2_rotation_0_360_20_rmsd_2.5
"""

import argparse
import os
import random
import pandas as pd
import numpy as np
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

# SCHRODINGER REPLACEMENT FUNCTIONS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('--group_name', type=str, default='', help='name of pose group subdir')
    parser.add_argument('--residue_cutoff', type=int, default=0, help='name of pose group subdir')
    parser.add_argument('--clash_cutoff', type=int, default=1, help='name of pose group subdir')
    args = parser.parse_args()
    random.seed(0)

    protein = 'P00797'
    target = '3own'
    start = '3d91'
    pair = '{}-to-{}'.format(target, start)
    protein_path = os.path.join(args.raw_root, protein)
    pair_path = os.path.join(protein_path, pair)
    pose_path = os.path.join(pair_path, args.group_name)
    incorrect_path = os.path.join(pose_path, 'subsample_incorrect')
    clash_path = os.path.join(incorrect_path, 'clash_data')
    prefix = 'pose_pred_data_'
    files = [f for f in os.listdir(clash_path) if f[:len(prefix)] == prefix]
    random.shuffle(files)
    files = files[:100]
    conformer_file = os.path.join(pair_path, "aligned_to_start_with_hydrogen_conformers.mae")
    conformers = list(structure.StructureReader(conformer_file))

    start_prot_file = os.path.join(pair_path, '{}_prot.mae'.format(start))
    start_prot = list(structure.StructureReader(start_prot_file))[0]

    start_prot_grid, start_origin = get_grid(start_prot)

    tolerable_poses = []
    intolerable_poses = []

    for file in files:
        df = pd.read_csv(os.path.join(clash_path, file))
        correct_df = df[df['pred_num_intolerable'] <= args.residue_cutoff]
        if len(correct_df) != 0:
            indices = [i for i in correct_df.index]
            indices = sorted(indices)
            i = random.choice(indices)
            tolerable_poses.append((file, i))

        incorrect_df = df[df['pred_num_intolerable'] > args.residue_cutoff]
        if len(incorrect_df) != 0:
            indices = [i for i in correct_df.index]
            indices = sorted(indices)
            i = random.choice(indices)
            intolerable_poses.append((file, i))

    for clash_cutoff in range(1, 5):
        num_without_clash = 0

        with structure.StructureWriter(os.path.join(clash_path,
                            'incorrect_tolerable_clash_revised_clash_cutoff_{}.mae'.format(clash_cutoff))) as correct:
            for file, i in tolerable_poses:
                df = pd.read_csv(os.path.join(clash_path, file))
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
                new_coords = rotate_structure(coords, math.radians(rot_x), math.radians(rot_y), math.radians(rot_z),
                                              conformer_center)

                # for clash features dictionary
                c.setXYZ(new_coords)

                start_clash = get_clash(c, start_prot_grid, start_origin)
                if start_clash <= clash_cutoff:
                    correct.append(c)
                    num_without_clash += 1

                c.setXYZ(old_coords)

        print('tolerable: clash cutoff = {}, num without clash = {}, num total = {}'.format(clash_cutoff,
                                                                            num_without_clash, len(tolerable_poses)))
        num_without_clash = 0

        with structure.StructureWriter(os.path.join(clash_path,
                        'incorrect_intolerable_clash_revised_clash_cutoff_{}.mae'.format(clash_cutoff))) as incorrect:
            for file, i in intolerable_poses:
                df = pd.read_csv(os.path.join(clash_path, file))
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
                new_coords = rotate_structure(coords, math.radians(rot_x), math.radians(rot_y), math.radians(rot_z),
                                              conformer_center)

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