"""
The purpose of this code is to create the cumulative frequency and bar graphs

It can be run on sherlock using
$ $SCHRODINGER/run python3 check_duplicate2.py /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
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
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('--protein', type=str, default='', help='protein name')
    parser.add_argument('--target', type=str, default='', help='target ligand name')
    parser.add_argument('--start', type=str, default='', help='start ligand name')
    parser.add_argument('--max_num_correct', type=int, default=390, help='maximum number of poses considered')
    parser.add_argument('--max_num_poses_considered', type=int, default=3900, help='maximum number of poses considered')
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
    parser.add_argument('--grid_n', type=int, default=75, help='number of grid_points processed in each job')
    parser.add_argument('--grid_search_step_size', type=int, default=2, help='step size between each grid point, in '
                                                                             'angstroms')
    args = parser.parse_args()

    random.seed(0)

    for protein, target, start in [('P00523', '4ybk', '2oiq'),
                                   ('P00519', '4twp', '5hu9'), ('P0DOX7', '6msy', '6mub')]:
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(args.raw_root, protein)
        pair_path = os.path.join(protein_path, pair)

        grid_size = get_grid_size(pair_path, target, start)
        group_name = 'test_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size)
        pose_path = os.path.join(pair_path, group_name)

        prefix = 'exhaustive_search_poses_'
        suffix = '.csv'

        incorrect = []

        for file in os.listdir(pose_path):
            if file[:len(prefix)] == prefix:
                df = pd.read_csv(os.path.join(pose_path, file))
                if len(df) != len(df.name.unique()):
                    stripped = file[len(prefix):-len(suffix)]
                    print(stripped)
                    vals = stripped.split('_')
                    grid_index = int(vals[0])
                    conformer_index = int(vals[1])
                    combined = (protein, target, start, grid_index, conformer_index)
                    if combined not in incorrect:
                        incorrect.append(combined)

        print(len(incorrect))
        print(incorrect)


if __name__=="__main__":
    main()