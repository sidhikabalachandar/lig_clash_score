"""
The purpose of this code is to create conformers

It can be run on sherlock using
$ $SCHRODINGER/run python3 search.py check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/sample/train/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw --group_name train_grid_6_1_rotation_0_360_20 --index 0 --n 1
"""

import argparse
import os
import schrodinger.structure as structure
from schrodinger.structutils.transform import get_centroid
import schrodinger.structutils.rmsd as rmsd
import schrodinger.structutils.interactions.steric_clash as steric_clash
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

# MAIN TASK FUNCTIONS


def check_pose(i, c, c_indices, target_lig, target_lig_indices, target_prot_grid, target_origin, target_prot,
               start_prot_grid, start_origin, start_prot, grid_loc, saved_dict, x, y, z):
    rmsd_val = rmsd.calculate_in_place_rmsd(c, c_indices, target_lig, target_lig_indices)
    start_clash = get_clash(c, start_prot_grid, start_origin)
    target_clash = get_clash(c, target_prot_grid, target_origin)
    schrod_start_clash = steric_clash.clash_volume(start_prot, struc2=c)
    schrod_target_clash = steric_clash.clash_volume(target_prot, struc2=c)
    name = '{}_{},{},{}_{},{},{}'.format(i, grid_loc[0], grid_loc[1], grid_loc[2], x, y, z)
    saved_dict['name'].append(name)
    saved_dict['conformer_index'].append(i)
    saved_dict['grid_loc_x'].append(grid_loc[0])
    saved_dict['grid_loc_y'].append(grid_loc[1])
    saved_dict['grid_loc_z'].append(grid_loc[2])
    saved_dict['rot_x'].append(x)
    saved_dict['rot_y'].append(y)
    saved_dict['rot_z'].append(z)
    saved_dict['start_clash'].append(start_clash)
    saved_dict['target_clash'].append(target_clash)
    saved_dict['schrod_start_clash'].append(schrod_start_clash)
    saved_dict['schrod_target_clash'].append(schrod_target_clash)
    saved_dict['rmsd'].append(rmsd_val)


def rotate_pose(i, c, c_indices, target_lig, target_lig_indices, target_prot_grid, target_origin, target_prot,
                start_prot_grid, start_origin, start_prot, grid_loc, coords, to_origin_matrix, from_origin_matrix,
                saved_dict, args):
    angles = [x for x in range(args.min_angle, args.max_angle + args.rotation_search_step_size,
                               args.rotation_search_step_size)]
    x = random.choice(angles)
    rot_matrix_x = get_rotation_matrix(X_AXIS, math.radians(x))
    y = random.choice(angles)
    rot_matrix_y = get_rotation_matrix(Y_AXIS, math.radians(y))
    z = random.choice(angles)
    rot_matrix_z = get_rotation_matrix(Z_AXIS, math.radians(z))


    # apply x,y,z rotation
    new_coords = rotate_structure(coords, from_origin_matrix, to_origin_matrix, rot_matrix_x,
                                  rot_matrix_y, rot_matrix_z)
    c.setXYZ(new_coords)

    check_pose(i, c, c_indices, target_lig, target_lig_indices, target_prot_grid, target_origin, target_prot,
               start_prot_grid, start_origin, start_prot, grid_loc, saved_dict, x, y, z)

    c.setXYZ(coords)


def search(protein, target, start, args):
    # important dirs
    pair = '{}-to-{}'.format(target, start)
    protein_path = os.path.join(args.raw_root, protein)
    pair_path = os.path.join(protein_path, pair)

    # ground truth lig
    target_lig_file = os.path.join(pair_path, 'ligand_poses', '{}_lig0.mae'.format(target))
    target_lig = list(structure.StructureReader(target_lig_file))[0]

    # get non hydrogen atom indices for rmsd
    target_lig_indices = [a.index for a in target_lig.atom if a.element != 'H']

    # prots
    start_prot_file = os.path.join(pair_path, '{}_prot.mae'.format(start))
    start_prot = list(structure.StructureReader(start_prot_file))[0]

    target_prot_file = os.path.join(pair_path, '{}_prot.mae'.format(target))
    target_prot = list(structure.StructureReader(target_prot_file))[0]

    # get conformers
    conformer_file = os.path.join(pair_path, "aligned_to_start_with_hydrogen_conformers.mae")
    conformers = list(structure.StructureReader(conformer_file))[:args.num_conformers]
    conformer_indices = [i for i in range(len(conformers))]

    # clash preprocessing
    start_prot_grid, start_origin = get_grid(start_prot)
    target_prot_grid, target_origin = get_grid(target_prot)

    # get grid
    grid_size = 6
    grid = []
    vals = set()
    for i in range(0, grid_size + 1):
        vals.add(i)
        vals.add(-i)
    for dx in vals:
        for dy in vals:
            for dz in vals:
                grid.append((dx, dy, dz))

    # get save location
    group_name = 'train_grid_{}_{}_rotation_{}_{}_{}'.format(grid_size, args.grid_search_step_size,
                                                                     args.min_angle, args.max_angle,
                                                                     args.rotation_search_step_size)
    pose_path = os.path.join(pair_path, group_name)
    if not os.path.exists(pose_path):
        os.mkdir(pose_path)

    if not os.path.exists(pose_path):
        os.mkdir(pose_path)
    saved_dict = {'name': [], 'conformer_index': [], 'grid_loc_x': [], 'grid_loc_y': [], 'grid_loc_z': [],
                  'rot_x': [], 'rot_y': [], 'rot_z': [], 'start_clash': [], 'target_clash': [],
                  'schrod_start_clash': [], 'schrod_target_clash': [], 'rmsd': []}

    # only save 100 poses
    for _ in range(100):
        # get conformer
        i = random.choice(conformer_indices)
        c = conformers[i]

        # get non hydrogen atom indices for rmsd
        c_indices = [a.index for a in c.atom if a.element != 'H']

        # random translation
        grid_loc = random.choice(grid)
        translate_structure(c, grid_loc[0], grid_loc[1], grid_loc[2])
        conformer_center = list(get_centroid(c))
        coords = c.getXYZ(copy=True)

        # rotation preprocessing
        displacement_vector = get_coords_array_from_list(conformer_center)
        to_origin_matrix = get_translation_matrix(-1 * displacement_vector)
        from_origin_matrix = get_translation_matrix(displacement_vector)

        rotate_pose(i, c, c_indices, target_lig, target_lig_indices, target_prot_grid, target_origin, target_prot,
                    start_prot_grid, start_origin, start_prot, grid_loc, coords, to_origin_matrix,
                    from_origin_matrix,
                    saved_dict, args)

        translate_structure(c, -grid_loc[0], -grid_loc[1], -grid_loc[2])

    df = pd.DataFrame.from_dict(saved_dict)
    print(os.path.join(pose_path, 'poses.csv'))
    df.to_csv(os.path.join(pose_path, 'poses.csv'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either align or search')
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('--num_conformers', type=int, default=2, help='maximum number of conformers considered')
    parser.add_argument('--n', type=int, default=10, help='number of grid_points processed in each job')
    parser.add_argument('--index', type=int, default=-1, help='grid point group index')
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
    parser.add_argument('--group_name', type=str, default='', help='name of pose group subdir')
    args = parser.parse_args()

    random.seed(0)

    if not os.path.exists(args.run_path):
        os.mkdir(args.run_path)

    if args.task == 'all':
        pairs = get_prots(args.docked_prot_file)
        grouped_pairs = group_files(args.n, pairs)
        counter = 0
        for i in range(len(grouped_pairs)):
            cmd = 'sbatch -p owners -t 0:20:00 -o {} --wrap="$SCHRODINGER/run python3 search.py group {} {} ' \
                  '{} --rotation_search_step_size {} --grid_size {} --n {} --num_conformers {} --index {}"'
            out_file_name = 'search_{}.out'.format(i)
            counter += 1
            os.system(cmd.format(os.path.join(args.run_path, out_file_name), args.docked_prot_file, args.run_path,
                                 args.raw_root, args.rotation_search_step_size, args.grid_size, args.n,
                                 args.num_conformers, i))

        print(counter)

    elif args.task == 'group':
        pairs = get_prots(args.docked_prot_file)
        grouped_pairs = group_files(args.n, pairs)
        for protein, target, start in grouped_pairs[args.index]:
            print(protein, target, start)
            search(protein, target, start, args)

    elif args.task == 'check':
        missing = []
        pairs = get_prots(args.docked_prot_file)
        for protein, target, start in pairs:
            pair = '{}-to-{}'.format(target, start)
            protein_path = os.path.join(args.raw_root, protein)
            pair_path = os.path.join(protein_path, pair)
            pose_path = os.path.join(pair_path, args.group_name)
            if not os.path.exists(pose_path):
                missing.append((protein, target, start))

        print('Missing: {}/{}'.format(len(missing), len(pairs)))
        print(missing)

    elif args.task == 'delete':
        pairs = get_prots(args.docked_prot_file)
        for protein, target, start in pairs:
            pair = '{}-to-{}'.format(target, start)
            protein_path = os.path.join(args.raw_root, protein)
            pair_path = os.path.join(protein_path, pair)
            pose_path = os.path.join(pair_path, args.group_name)
            os.system("rm -rf {}".format(pose_path))


if __name__ == "__main__":
    main()
