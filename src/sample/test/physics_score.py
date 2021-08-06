"""
The purpose of this code is to create the cumulative frequency and bar graphs

It can be run on sherlock using
$ $SCHRODINGER/run python3 cum_freq.py /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw --protein P02829 --target 2weq --start 2yge
"""

import argparse
import os
import pandas as pd
import schrodinger.structure as structure
import random
from schrodinger.structutils.transform import get_centroid
import math
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
    args = parser.parse_args()

    random.seed(0)

    pair = '{}-to-{}'.format(args.target, args.start)
    protein_path = os.path.join(args.raw_root, args.protein)
    pair_path = os.path.join(protein_path, pair)

    grid_size = get_grid_size(pair_path, args.target, args.start)
    group_name = 'test_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size)
    pose_path = os.path.join(pair_path, group_name)

    conformer_file = os.path.join(pair_path, "aligned_to_start_with_hydrogen_conformers.mae")
    conformers = list(structure.StructureReader(conformer_file))

    df = pd.read_csv(os.path.join(pose_path, 'poses_after_advanced_filter.csv'))

    name = '10_0,0,0_0,20,0'
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

    protein_file = os.path.join(pair_path, '{}_prot.mae'.format(args.start))
    prot_s = list(structure.StructureReader(protein_file))[0]

    ligand_coord = new_coords
    ligand_charge = np.array([a.partial_charge for a in c.atom])
    target_coord = prot_s.getXYZ(copy=True)
    target_charge = np.array([a.partial_charge for a in prot_s.atom])
    ligand_atom_type = [a.element for a in c.atom]
    target_atom_type = [a.element for a in prot_s.atom]

    print(physics_score(ligand_coord, ligand_charge, target_coord, target_charge, ligand_atom_type, target_atom_type))

    c.setXYZ(old_coords)


if __name__=="__main__":
    main()