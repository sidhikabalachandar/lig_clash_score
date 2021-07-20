"""
The purpose of this code is to create conformers

It can be run on sherlock using
$ $SCHRODINGER/run python3 test_search.py /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/sample/train/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
"""

import argparse
import os
import schrodinger.structure as structure
from schrodinger.structutils.transform import get_centroid
import schrodinger.structutils.rmsd as rmsd
import random
import math
import pandas as pd
import sys
sys.path.insert(1, '../util')
from util import *
from prot_util import *
from schrod_replacement_util import *

X_AXIS = [1.0, 0.0, 0.0]  # x-axis unit vector
Y_AXIS = [0.0, 1.0, 0.0]  # y-axis unit vector
Z_AXIS = [0.0, 0.0, 1.0]  # z-axis unit vector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('--num_conformers', type=int, default=2, help='maximum number of conformers considered')
    parser.add_argument('--grid_size', type=int, default=1, help='grid size in positive and negative x, y, z '
                                                                 'directions')
    parser.add_argument('--grid_search_step_size', type=int, default=1, help='step size between each grid point, in '
                                                                             'angstroms')
    parser.add_argument('--rotation_search_step_size', type=int, default=20, help='step size between each angle '
                                                                                 'checked, in degrees')
    parser.add_argument('--min_angle', type=int, default=0, help='min angle of rotation in degrees')
    parser.add_argument('--max_angle', type=int, default=360, help='min angle of rotation in degrees')
    parser.add_argument('--start_clash_cutoff', type=int, default=100, help='clash cutoff between start protein and '
                                                                            'ligand pose')
    args = parser.parse_args()

    random.seed(0)

    if not os.path.exists(args.run_path):
        os.mkdir(args.run_path)

    pairs = get_prots(args.docked_prot_file)
    counter = 0
    pair_index = random.choice([i for i in range(len(pairs))])
    protein, target, start = pairs[pair_index]
    print(protein, target, start)
    pair = '{}-to-{}'.format(target, start)
    protein_path = os.path.join(args.raw_root, protein)
    pair_path = os.path.join(protein_path, pair)

    # get grid
    grid_size = 6

    group_name = 'train_grid_{}_{}_rotation_{}_{}_{}'.format(grid_size, args.grid_search_step_size,
                                                                     args.min_angle, args.max_angle,
                                                                     args.rotation_search_step_size)
    pose_path = os.path.join(pair_path, group_name)
    pose_file = os.path.join(pose_path, 'poses.csv')

    # get conformers
    conformer_file = os.path.join(pair_path, "aligned_to_start_with_hydrogen_conformers.mae")
    conformers = list(structure.StructureReader(conformer_file))[:args.num_conformers]
    if not os.path.exists(pose_file):
        cmd = 'sbatch -p owners -t 0:20:00 -o {} --wrap="$SCHRODINGER/run python3 search.py group {} {} ' \
              '{} --rotation_search_step_size {} --grid_size {} --n 1 --num_conformers {} --index {}"'
        out_file_name = 'search_{}.out'.format(i)
        counter += 1
        os.system(cmd.format(os.path.join(args.run_path, out_file_name), args.docked_prot_file, args.run_path,
                             args.raw_root, args.rotation_search_step_size, args.grid_size, 1, args.num_conformers,
                             pair_index))

    df = pd.read_csv(pose_file)
    tolerable_indices = [i for i in df.index]
    pose_index = random.choice(tolerable_indices)

    conformer_index = df.loc[[pose_index]]['conformer_index'].iloc[0]
    c = conformers[conformer_index]
    old_coords = c.getXYZ(copy=True)
    grid_loc_x = df.loc[[pose_index]]['grid_loc_x'].iloc[0]
    grid_loc_y = df.loc[[pose_index]]['grid_loc_y'].iloc[0]
    grid_loc_z = df.loc[[pose_index]]['grid_loc_z'].iloc[0]
    translate_structure(c, grid_loc_x, grid_loc_y, grid_loc_z)
    conformer_center = list(get_centroid(c))
    coords = c.getXYZ(copy=True)
    rot_x = df.loc[[pose_index]]['rot_x'].iloc[0]
    rot_y = df.loc[[pose_index]]['rot_y'].iloc[0]
    rot_z = df.loc[[pose_index]]['rot_z'].iloc[0]

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

    df_start_clash = df.loc[[pose_index]]['start_clash'].iloc[0]
    df_target_clash = df.loc[[pose_index]]['target_clash'].iloc[0]
    df_rmsd = df.loc[[pose_index]]['rmsd'].iloc[0]

    # prots
    start_prot_file = os.path.join(pair_path, '{}_prot.mae'.format(start))
    start_prot = list(structure.StructureReader(start_prot_file))[0]

    target_prot_file = os.path.join(pair_path, '{}_prot.mae'.format(target))
    target_prot = list(structure.StructureReader(target_prot_file))[0]

    # clash preprocessing
    start_prot_grid, start_origin = get_grid(start_prot)
    target_prot_grid, target_origin = get_grid(target_prot)

    c_indices = [a.index for a in c.atom if a.element != 'H']

    # ground truth lig
    target_lig_file = os.path.join(pair_path, 'ligand_poses', '{}_lig0.mae'.format(target))
    target_lig = list(structure.StructureReader(target_lig_file))[0]

    # get non hydrogen atom indices for rmsd
    target_lig_indices = [a.index for a in target_lig.atom if a.element != 'H']

    rmsd_val = rmsd.calculate_in_place_rmsd(c, c_indices, target_lig, target_lig_indices)
    start_clash = get_clash(c, start_prot_grid, start_origin)
    target_clash = get_clash(c, target_prot_grid, target_origin)

    assert(df_start_clash == start_clash)
    assert (df_target_clash == target_clash)
    assert (df_rmsd == rmsd_val)
    print('All correct!')

    c.setXYZ(old_coords)




if __name__ == "__main__":
    main()
