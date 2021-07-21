"""
The purpose of this code is to first create the raw directory folder and include the following files
starting protein receptor
starting ligand
target ligand
glide pose viewer file

Then the top glide poses are added

Then the decoys are created

It can be run on sherlock using
$ $SCHRODINGER/run python3 test.py /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw --group_name exhaustive_grid_4_2_rotation_0_360_20_rmsd_2.5
"""

import argparse
import os
import random
import pandas as pd
import numpy as np
import math
import pickle
import statistics
from tqdm import tqdm
import schrodinger.structure as structure
from schrodinger.structutils.transform import get_centroid
import schrodinger.structutils.interactions.steric_clash as steric_clash
import time
import scipy.spatial

X_AXIS = [1.0, 0.0, 0.0]  # x-axis unit vector
Y_AXIS = [0.0, 1.0, 0.0]  # y-axis unit vector
Z_AXIS = [0.0, 0.0, 1.0]  # z-axis unit vector

# HELPER FUNCTIONS

def get_grid(s):
    """
    Generate the 3d grid from coordinate format.
    Args:
        df (pd.DataFrame):
            region to generate grid for.
        center (3x3 np.array):
            center of the grid.
        rot_mat (3x3 np.array):
            rotation matrix to apply to region before putting in grid.
    Returns:
        4-d numpy array representing an occupancy grid where last dimension
        is atom channel.  First 3 dimension are of size radius_ang * 2 + 1.
    """
    at = s.getXYZ(copy=True)
    # at = at * 2
    at = (np.around(at - 0.5)).astype(np.int16)
    origin = np.full((3), np.amin(at))
    at = at - origin
    dim = np.amax(at) * 2
    grid = np.zeros((dim, dim, dim))
    np.add.at(grid, (at[:, 0], at[:, 1], at[:, 2]), 1)

    return grid, origin


def get_clash(s, grid, origin):
    at = s.getXYZ(copy=True)
    # at = at * 2
    at = (np.around(at - 0.5)).astype(np.int16)
    at = at - origin
    return np.sum(grid[at[:, 0], at[:, 1], at[:, 2]])


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

    for clash_cutoff in [3, 3, 3, 3, 3]:
        num_without_clash = 0
        num_total = 0

        for file in files:
            df = pd.read_csv(os.path.join(clash_path, file))
            correct_df = df[df['pred_num_intolerable'] <= args.residue_cutoff]
            if len(correct_df) != 0:
                num_total += 1
                indices = [i for i in correct_df.index]
                i = random.choice(indices)
                conformer_index = df.loc[[i]]['conformer_index'].iloc[0]
                c = conformers[conformer_index]

                start_clash = get_clash(c, start_prot_grid, start_origin)
                if start_clash <= clash_cutoff:
                    num_without_clash += 1

        print('tolerable: clash cutoff = {}, num without clash = {}, num total = {}'.format(clash_cutoff,
                                                                                        num_without_clash, num_total))


if __name__ == "__main__":
    main()
