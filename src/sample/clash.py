"""
The purpose of this code is to create conformers

It can be run on sherlock using
$ $SCHRODINGER/run python3 clash.py
"""

import argparse
import os
import schrodinger.structure as structure
import schrodinger.structutils.build as build
from schrodinger.structutils.transform import get_centroid
import schrodinger.structutils.transform as transform
import schrodinger.structutils.interactions.steric_clash as steric_clash
import schrodinger.structutils.rmsd as rmsd
import random
import pickle
import scipy.spatial
import time
import math
import numpy as np
import statistics
import pandas as pd
from tqdm import tqdm



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
    at = (np.around(at - 0.5)).astype(np.int16)
    origin = np.full((3), np.amin(at))
    at = at - origin
    dim = np.amax(at) + 1
    grid = np.zeros((dim, dim, dim))
    np.add.at(grid, (at[:, 0], at[:, 1], at[:, 2]), 1)

    return grid, origin


def get_clash(s, grid, origin):
    at = s.getXYZ(copy=True)
    at = (np.around(at - 0.5)).astype(np.int16)
    at = at - origin
    return np.sum(grid[at[:, 0], at[:, 1], at[:, 2]])

def main():
    raw_root = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw'
    protein = 'P02829'
    target = '2fxs'
    start = '2weq'
    pair = '{}-to-{}'.format(target, start)
    protein_path = os.path.join(raw_root, protein)
    pair_path = os.path.join(protein_path, pair)
    target_lig_file = os.path.join(pair_path, 'ligand_poses', '{}_lig0.mae'.format(target))
    target_lig = list(structure.StructureReader(target_lig_file))[0]
    start_prot_file = os.path.join(pair_path, '{}_prot.mae'.format(start))
    start_prot = list(structure.StructureReader(start_prot_file))[0]

    grid, origin = get_grid(start_prot)
    start = time.time()
    custom_clash_vol = get_clash(target_lig, grid, origin)
    end = time.time()
    custom_time = end - start

    print(custom_clash_vol, custom_time)

    start = time.time()
    schrod_clash_vol = steric_clash.clash_volume(start_prot, struc2=target_lig)
    end = time.time()
    schrod_time = end - start

    print(schrod_clash_vol, schrod_time)


if __name__ == "__main__":
    main()
