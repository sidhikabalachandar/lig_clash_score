"""
The purpose of this code is to create conformers

It can be run on sherlock using
$ $SCHRODINGER/run python3 search.py check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits/search_test_incorrect_glide_index.txt /home/users/sidhikab/lig_clash_score/src/sample/test/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw --protein P11838 --target 3wz6 --start 1gvx --grid_index 0 --conformer_index 9
"""

import argparse
import os
import schrodinger.structure as structure
from schrodinger.structutils.transform import get_centroid
import schrodinger.structutils.rmsd as rmsd
import random
import scipy.spatial
import time
import math
import numpy as np
import pandas as pd

X_AXIS = [1.0, 0.0, 0.0]  # x-axis unit vector
Y_AXIS = [0.0, 1.0, 0.0]  # y-axis unit vector
Z_AXIS = [0.0, 0.0, 1.0]  # z-axis unit vector

# HELPER FUNCTIONS


def get_prots(args):
    """
    gets list of all protein, target ligands, and starting ligands in the index file
    :param docked_prot_file: (string) file listing proteins to process
    :return: process (list) list of all protein, target ligands, and starting ligands to process
    """
    process = []
    with open(args.docked_prot_file) as fp:
        for line in fp:
            if line[0] == '#':
                continue
            protein, target, start = line.strip().split()
            process.append((protein, target, start))

    return process


def group_grid(n, grid_size, step):
    grid = []
    vals = set()
    for i in range(0, grid_size + 1, step):
        vals.add(i)
        vals.add(-i)
    for dx in vals:
        for dy in vals:
            for dz in vals:
                grid.append((dx, dy, dz))

    grouped_files = []

    for i in range(0, len(grid), n):
        grouped_files += [grid[i: i + n]]

    return grouped_files


def get_pocket_res(protein, ligand, dist):
    """
    Given a co-crystallized protein and ligand, extract residues within specified distance of ligand.

    Args:
        protein (Biopython Structure object): receptor protein
        ligand (RDKit Mol object): co-crystallized ligand
        dist (float): distance cutoff for defining binding site

    Returns:
        key_residues (set of Biopython Residue objects): set of key binding site residues
    """
    # get protein coordinates
    prot_coords = []
    for m in list(protein.molecule):
        for r in list(m.residue):
            for a in list(r.atom):
                prot_coords.append(a.xyz)

    # get ligand coordinates
    lig_coords = []
    for m in list(ligand.molecule):
        for r in list(m.residue):
            for a in list(r.atom):
                lig_coords.append(a.xyz)

    kd_tree = scipy.spatial.KDTree(prot_coords)
    key_pts = kd_tree.query_ball_point(lig_coords, r=dist, p=2.0)
    key_pts = set([k for l in key_pts for k in l])

    remove = [i for i in protein.getAtomIndices() if i not in key_pts]
    protein.deleteAtoms(remove)


def group_files(n, process):
    """
    groups pairs into sublists of size n
    :param n: (int) sublist size
    :param process: (list) list of pairs to process
    :return: grouped_files (list) list of sublists of pairs
    """
    grouped_files = []

    for i in range(0, len(process), n):
        grouped_files += [process[i: i + n]]

    return grouped_files


def get_grid_size(pair_path, target, start):
    target_lig_file = os.path.join(pair_path, 'ligand_poses', '{}_lig0.mae'.format(target))
    target_lig = list(structure.StructureReader(target_lig_file))[0]
    target_center = get_centroid(target_lig)

    start_lig_file = os.path.join(pair_path, '{}_lig.mae'.format(start))
    start_lig = list(structure.StructureReader(start_lig_file))[0]
    start_center = get_centroid(start_lig)

    dist = np.sqrt((target_center[0] - start_center[0]) ** 2 +
                   (target_center[1] - start_center[1]) ** 2 +
                   (target_center[2] - start_center[2]) ** 2)

    grid_size = int(dist + 1)
    if grid_size % 2 == 1:
        grid_size += 1
    return grid_size

# SCHRODINGER REPLACEMENT FUNCTIONS


def get_translation_matrix(trans):
    """
    Returns a 4x4 numpy array representing a translation matrix from
    a 3-element list.

    trans (list)
        3-element list (x,y,z).
    """

    trans_matrix = np.identity(4, 'd')  # four floats
    trans_matrix[0][3] = float(trans[0])
    trans_matrix[1][3] = float(trans[1])
    trans_matrix[2][3] = float(trans[2])
    return trans_matrix


def transform_structure(coords, matrix):
    """
    Transforms atom coordinates of the structure using a 4x4
    transformation matrix.

    st (structure.Structure)

    matrix (numpy.array)
        4x4 numpy array representation of transformation matrix.

    """

    # Modifying this array will directly alter the actual coordinates:
    atom_xyz_array = np.array(coords)
    num_atoms = len(atom_xyz_array)
    ones = np.ones((num_atoms, 1))
    atom_xyz_array = np.concatenate((atom_xyz_array, ones), axis=1)
    atom_xyz_array = np.resize(atom_xyz_array, (num_atoms, 4, 1))
    atom_xyz_array = np.matmul(matrix, atom_xyz_array)
    atom_xyz_array = np.resize(atom_xyz_array, (num_atoms, 4))
    atom_xyz_array = atom_xyz_array[:, 0:3]
    return atom_xyz_array


def translate_structure(st, x, y, z):
    trans_matrix = get_translation_matrix([x, y, z])
    atom_xyz_array = transform_structure(st.getXYZ(copy=True), trans_matrix)
    st.setXYZ(atom_xyz_array)


def get_coords_array_from_list(coords_list):
    """
    Returns coordinates as a 4-element numpy array: (x,y,z,0.0).

    coords_list (list or array)
        3 elements: x, y, z.

    """

    coords = np.zeros((4), 'd')  # four floats
    coords[0] = coords_list[0]
    coords[1] = coords_list[1]
    coords[2] = coords_list[2]
    return coords


def get_rotation_matrix(axis, angle):
    """
    Returns a 4x4 numpy array representing a right-handed rotation
    matrix about the specified axis running through the origin by some angle

    axis (vector)
        Normalized (unit) vector for the axis around which to rotate.
        Can be one of predefined axis: X_AXIS, Y_AXIS, Z_AXIS, or arbitrary
        axis.

    angle (float)
        Angle, in radians, about which to rotate the structure about
        the axis.
    """

    # From: http://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/index.htm
    #
    # Rotation matrix =
    #
    # t*x*x + c       t*x*y - z*s     t*x*z + y*s
    # t*x*y + z*s     t*y*y + c       t*y*z - x*s
    # t*x*z - y*s     t*y*z + x*s     t*z*z + c
    #
    # where,
    #
    # * c = cos(angle)
    # * s = sin(angle)
    # * t = 1 - c
    # * x = normalised x portion of the axis vector
    # * y = normalised y portion of the axis vector
    # * z = normalised z portion of the axis vector

    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c
    x = axis[0]
    y = axis[1]
    z = axis[2]

    rot_matrix = np.identity(4, 'd')  # four floats
    rot_matrix[0] = [t * x * x + c, t * x * y - z * s, t * x * z + y * s, 0.0]
    rot_matrix[1] = [t * x * y + z * s, t * y * y + c, t * y * z - x * s, 0.0]
    rot_matrix[2] = [t * x * z - y * s, t * y * z + x * s, t * z * z + c, 0.0]

    return rot_matrix


def rotate_structure(coords, from_origin_matrix, to_origin_matrix, rot_matrix_x, rot_matrix_y, rot_matrix_z):
    """
    Rotates the structure about x axis, then y axis, then z axis.

    st (structure.Structure)

    x_angle (float)
        Angle, in radians, about x to right-hand rotate.

    y_angle (float)
        Angle, in radians, about y to right-hand rotate.

    z_angle (float)
        Angle, in radians, about z to right-hand rotate.

    rot_center (list)
        Cartesian coordinates (x, y, z) for the center of rotation.
        By default, rotation happens about the origin (0, 0, 0)

    """
    combined_rot_matrix = np.matmul(np.matmul(np.matmul(np.matmul(from_origin_matrix, rot_matrix_z), rot_matrix_y),
                                              rot_matrix_x), to_origin_matrix)
    return transform_structure(coords, combined_rot_matrix)


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
    at = (np.around(at - 0.5)).astype(np.int16)
    at = at - origin
    return np.sum(grid[at[:, 0], at[:, 1], at[:, 2]])

# MAIN TASK FUNCTIONS


def get_one_pose(pair_path, args):
    target_lig_file = os.path.join(pair_path, 'ligand_poses', '{}_lig0.mae'.format(args.target))
    target_lig = list(structure.StructureReader(target_lig_file))[0]
    target_lig_indices = [a.index for a in target_lig.atom if a.element != 'H']
    conformer_file = os.path.join(pair_path, "aligned_to_start_with_hydrogen_conformers.mae")
    conformers = list(structure.StructureReader(conformer_file))[:300]

    conformer_index = 28
    grid_loc_x = 2
    grid_loc_y = 4
    grid_loc_z = 0
    rot_x = 80
    rot_y = 40
    rot_z = 320

    c = conformers[conformer_index]
    c_indices = [a.index for a in c.atom if a.element != 'H']

    translate_structure(c, grid_loc_x, grid_loc_y, grid_loc_z)
    conformer_center = list(get_centroid(c))
    coords = c.getXYZ(copy=True)

    # rotation preprocessing
    displacement_vector = get_coords_array_from_list(conformer_center)
    to_origin_matrix = get_translation_matrix(-1 * displacement_vector)
    from_origin_matrix = get_translation_matrix(displacement_vector)

    rot_matrix_x = get_rotation_matrix(X_AXIS, math.radians(rot_x))
    rot_matrix_y = get_rotation_matrix(Y_AXIS, math.radians(rot_y))
    rot_matrix_z = get_rotation_matrix(Z_AXIS, math.radians(rot_z))

    # apply x,y,z rotation
    new_coords = rotate_structure(coords, from_origin_matrix, to_origin_matrix, rot_matrix_x,
                                  rot_matrix_y, rot_matrix_z)
    c.setXYZ(new_coords)

    # check if pose correct
    rmsd_val = rmsd.calculate_in_place_rmsd(c, c_indices, target_lig, target_lig_indices)
    print('one pose without H', rmsd_val)
    rmsd_val = rmsd.calculate_in_place_rmsd(c, c.getAtomIndices(), target_lig, target_lig.getAtomIndices())
    print('one pose with H', rmsd_val)


def check_pose(num_poses_searched, i, grid_loc, start_prot_grid, start_origin, c_indices, target_lig,
               target_lig_indices, args, num_correct, num_after_simple_filter, num_correct_after_simple_filter,
               target_prot_grid, target_origin, saved_dict, c, x, y, z, pair_path):
    # look at potential pose
    num_poses_searched += 1

    # check simple filter
    start_clash = get_clash(c, start_prot_grid, start_origin)

    # check if pose correct
    rmsd_val = rmsd.calculate_in_place_rmsd(c, c_indices, target_lig, target_lig_indices)
    # name = '{}_{},{},{}_{},{},{}'.format(i, grid_loc[0], grid_loc[1], grid_loc[2], x, y, z)
    # print(name)
    # if name == '28_2,4,0_80,40,320':
    #     get_one_pose(pair_path, args)
    #     print(rmsd_val)
    #     assert(1 == 2)
    if rmsd_val < args.rmsd_cutoff:
        num_correct += 1

    if start_clash == 0:
        num_after_simple_filter += 1
        if rmsd_val < args.rmsd_cutoff:
            num_correct_after_simple_filter += 1
        # save info for pose
        target_clash = get_clash(c, target_prot_grid, target_origin)
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
        saved_dict['rmsd'].append(rmsd_val)

    return num_poses_searched, num_correct, num_after_simple_filter, num_correct_after_simple_filter


def rotate_pose(args, coords, from_origin_matrix, to_origin_matrix, c,
                num_poses_searched, i, grid_loc, start_prot_grid, start_origin, c_indices, target_lig,
                target_lig_indices, num_correct, num_after_simple_filter, num_correct_after_simple_filter,
                target_prot_grid, target_origin, saved_dict, pair_path):
    for x in range(args.min_angle, args.max_angle + args.rotation_search_step_size, args.rotation_search_step_size):
        # rotation preprocessing
        rot_matrix_x = get_rotation_matrix(X_AXIS, math.radians(x))

        for y in range(args.min_angle, args.max_angle + args.rotation_search_step_size, args.rotation_search_step_size):
            # rotation preprocessing
            rot_matrix_y = get_rotation_matrix(Y_AXIS, math.radians(y))

            for z in range(args.min_angle, args.max_angle + args.rotation_search_step_size, args.rotation_search_step_size):
                # rotation preprocessing
                rot_matrix_z = get_rotation_matrix(Z_AXIS, math.radians(z))

                # apply x,y,z rotation
                new_coords = rotate_structure(coords, from_origin_matrix, to_origin_matrix, rot_matrix_x,
                                              rot_matrix_y, rot_matrix_z)
                c.setXYZ(new_coords)

                num_poses_searched, num_correct, num_after_simple_filter, num_correct_after_simple_filter = \
                    check_pose(num_poses_searched, i, grid_loc, start_prot_grid, start_origin, c_indices, target_lig,
                               target_lig_indices, args, num_correct, num_after_simple_filter,
                               num_correct_after_simple_filter, target_prot_grid, target_origin, saved_dict, c, x, y, z, pair_path)

                c.setXYZ(coords)

    return num_poses_searched, num_correct, num_after_simple_filter, num_correct_after_simple_filter


def search(args):
    # important dirs
    pair = '{}-to-{}'.format(args.target, args.start)
    protein_path = os.path.join(args.raw_root, args.protein)
    pair_path = os.path.join(protein_path, pair)

    target_lig_file = os.path.join(pair_path, 'ligand_poses', '{}_lig0.mae'.format(args.target))
    target_lig = list(structure.StructureReader(target_lig_file))[0]
    # get non hydrogen atom indices for rmsd
    target_lig_indices = [a.index for a in target_lig.atom if a.element != 'H']

    start_prot_file = os.path.join(pair_path, '{}_prot.mae'.format(args.start))
    start_prot = list(structure.StructureReader(start_prot_file))[0]

    target_prot_file = os.path.join(pair_path, '{}_prot.mae'.format(args.target))
    target_prot = list(structure.StructureReader(target_prot_file))[0]

    # get conformers
    conformer_file = os.path.join(pair_path, "aligned_to_start_with_hydrogen_conformers.mae")
    conformers = list(structure.StructureReader(conformer_file))[:args.num_conformers]
    conformer_indices = [i for i in range(len(conformers))]
    grouped_conformer_indices = group_files(args.conformer_n, conformer_indices)
    conformer_group_indices = grouped_conformer_indices[args.conformer_index]

    # get grid
    grid_size = get_grid_size(pair_path, args.target, args.start)
    grouped_files = group_grid(args.grid_n, grid_size, args.grid_search_step_size)
    grid = grouped_files[args.grid_index]

    # get save location
    group_name = 'exhaustive_grid_{}_{}_rotation_{}_{}_{}_rmsd_{}'.format(grid_size,
                 args.grid_search_step_size, args.min_angle, args.max_angle, args.rotation_search_step_size, args.rmsd_cutoff)
    pose_path = os.path.join(pair_path, group_name)
    if not os.path.exists(pose_path):
        os.mkdir(pose_path)

    # clash preprocessing
    start_prot_grid, start_origin = get_grid(start_prot)
    target_prot_grid, target_origin = get_grid(target_prot)

    data_dict = {'protein': [], 'target': [], 'start': [], 'num_conformers': [], 'num_grid_locs': [],
                 'num_poses_searched': [], 'num_correct': [], 'num_after_simple_filter': [],
                 'num_correct_after_simple_filter': [], 'time_elapsed': []}

    num_poses_searched = 0
    num_correct = 0
    num_after_simple_filter = 0
    num_correct_after_simple_filter = 0

    saved_dict = {'name': [], 'conformer_index': [], 'grid_loc_x': [], 'grid_loc_y': [], 'grid_loc_z': [],
                  'rot_x': [], 'rot_y': [], 'rot_z': [], 'start_clash': [], 'target_clash': [], 'rmsd': []}

    with open(os.path.join(pose_path, 'exhaustive_search_poses_{}_{}.csv'.format(args.grid_index, args.conformer_index)),
              'w') as f:
        df = pd.DataFrame.from_dict(saved_dict)
        df.to_csv(f)

    decoy_start_time = time.time()

    for i in conformer_group_indices:
        c = conformers[i]
        saved_dict = {'name': [], 'conformer_index': [], 'grid_loc_x': [], 'grid_loc_y': [], 'grid_loc_z': [],
                      'rot_x': [], 'rot_y': [], 'rot_z': [], 'start_clash': [], 'target_clash': [], 'rmsd': []}

        # get non hydrogen atom indices for rmsd
        c_indices = [a.index for a in c.atom if a.element != 'H']

        for grid_loc in grid:
            # apply grid_loc translation
            translate_structure(c, grid_loc[0], grid_loc[1], grid_loc[2])
            conformer_center = list(get_centroid(c))
            coords = c.getXYZ(copy=True)

            # rotation preprocessing
            displacement_vector = get_coords_array_from_list(conformer_center)
            to_origin_matrix = get_translation_matrix(-1 * displacement_vector)
            from_origin_matrix = get_translation_matrix(displacement_vector)

            num_poses_searched, num_correct, num_after_simple_filter, num_correct_after_simple_filter = \
                rotate_pose(args, coords, from_origin_matrix, to_origin_matrix, c, num_poses_searched, i, grid_loc,
                            start_prot_grid, start_origin, c_indices, target_lig, target_lig_indices, num_correct,
                            num_after_simple_filter, num_correct_after_simple_filter, target_prot_grid, target_origin,
                            saved_dict, pair_path)

            translate_structure(c, -grid_loc[0], -grid_loc[1], -grid_loc[2])

        with open(os.path.join(pose_path, 'exhaustive_search_poses_{}_{}.csv'.format(
                args.grid_index, args.conformer_index)), 'a') as f:
            df = pd.DataFrame.from_dict(saved_dict)
            df.to_csv(f, header=False)

    # save info for grid_loc
    decoy_end_time = time.time()
    data_dict['protein'].append(args.protein)
    data_dict['target'].append(args.target)
    data_dict['start'].append(args.start)
    data_dict['num_conformers'].append(len(conformer_group_indices))
    data_dict['num_grid_locs'].append(len(grid))
    data_dict['num_poses_searched'].append(num_poses_searched)
    data_dict['num_correct'].append(num_correct)
    data_dict['num_after_simple_filter'].append(num_after_simple_filter)
    data_dict['num_correct_after_simple_filter'].append(num_correct_after_simple_filter)
    data_dict['time_elapsed'].append(decoy_end_time - decoy_start_time)

    df = pd.DataFrame.from_dict(data_dict)
    df.to_csv(os.path.join(pose_path, 'exhaustive_search_info_{}_{}.csv'.format(args.grid_index, args.conformer_index)))


def check_search(pairs, raw_root):
    unfinished = []
    for protein, target, start in pairs:
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, pair)
        if not os.path.exists(os.path.join(pair_path, 'exhaustive_search_poses.csv')):
            unfinished.append((protein, target, start))

    print("Missing:", len(unfinished), "/", len(pairs))
    print(unfinished)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either align or search')
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('--num_conformers', type=int, default=300, help='maximum number of conformers considered')
    parser.add_argument('--num_pairs', type=int, default=10, help='number of protein-ligand pairs considered')
    parser.add_argument('--grid_n', type=int, default=75, help='number of grid_points processed in each job')
    parser.add_argument('--grid_size', type=int, default=1, help='grid size in positive and negative x, y, z '
                                                                 'directions')
    parser.add_argument('--grid_file', type=str, default='', help='pickle file with grid data dictionary')
    parser.add_argument('--grid_index', type=int, default=-1, help='grid point group index')
    parser.add_argument('--grid_search_step_size', type=int, default=2, help='step size between each grid point, in '
                                                                             'angstroms')
    parser.add_argument('--rotation_search_step_size', type=int, default=20, help='step size between each angle '
                                                                                 'checked, in degrees')
    parser.add_argument('--min_angle', type=int, default=0, help='min angle of rotation in degrees')
    parser.add_argument('--max_angle', type=int, default=360, help='min angle of rotation in degrees')
    parser.add_argument('--conformer_n', type=int, default=3, help='number of conformers processed in each job')
    parser.add_argument('--conformer_index', type=int, default=-1, help='number of grid_points processed in each job')
    parser.add_argument('--rmsd_cutoff', type=int, default=2.5, help='rmsd accuracy cutoff between predicted ligand pose '
                                                                   'and true ligand pose')
    parser.add_argument('--start_clash_cutoff', type=int, default=100, help='clash cutoff between start protein and '
                                                                            'ligand pose')
    parser.add_argument('--protein', type=str, default='', help='name of protein')
    parser.add_argument('--target', type=str, default='', help='name of target ligand')
    parser.add_argument('--start', type=str, default='', help='name of start ligand')
    args = parser.parse_args()

    random.seed(0)

    if not os.path.exists(args.run_path):
        os.mkdir(args.run_path)

    if args.task == 'all':
        pairs = get_prots(args)
        random.shuffle(pairs)
        counter = 0
        # for protein, target, start in pairs[:5]:
        pairs = [('Q86WV6', '4loi', '4qxo', 0, 0), ('Q86WV6', '4loi', '4qxo', 0, 1), ('Q86WV6', '4loi', '4qxo', 0, 4), ('Q86WV6', '4loi', '4qxo', 1, 0), ('Q86WV6', '4loi', '4qxo', 1, 1)]
        for protein, target, start, i, j in pairs:
            pair = '{}-to-{}'.format(target, start)
            protein_path = os.path.join(args.raw_root, protein)
            pair_path = os.path.join(protein_path, pair)
            conformer_file = os.path.join(pair_path, "aligned_to_start_with_hydrogen_conformers.mae")
            conformers = list(structure.StructureReader(conformer_file))[:args.num_conformers]
            grouped_conformers = group_files(args.conformer_n, conformers)

            grid_size = get_grid_size(pair_path, target, start)
            grouped_grid_locs = group_grid(args.grid_n, grid_size, 2)

            print('protein: {}, target: {}, start: {}, num_conformers: {}, grid_size {}'.format(
                protein, target, start, len(conformers), grid_size))

            # for i in range(len(grouped_grid_locs)):
            #     for j in range(len(grouped_conformers)):
            cmd = 'sbatch -p owners -t 0:20:00 -o {} --wrap="$SCHRODINGER/run python3 search.py group {} {} {} ' \
                  '--rotation_search_step_size {} --grid_size {} --grid_n {} --num_conformers {} ' \
                  '--conformer_n {} --grid_index {} --conformer_index {} --protein {} --target {} --start {}"'
            out_file_name = 'search_{}_{}_{}_{}_{}.out'.format(protein, target, start, i, j)
            counter += 1
            os.system(
                cmd.format(os.path.join(args.run_path, out_file_name), args.docked_prot_file, args.run_path,
                           args.raw_root, args.rotation_search_step_size, args.grid_size, args.grid_n,
                           args.num_conformers, args.conformer_n, i, j, protein, target, start))

        print(counter)

    elif args.task == 'group':
        search(args)

    elif args.task == 'check':
        pairs = get_prots(args)
        random.shuffle(pairs)
        missing = []
        counter = 0
        for protein, target, start in pairs[:5]:
            pair = '{}-to-{}'.format(target, start)
            protein_path = os.path.join(args.raw_root, protein)
            pair_path = os.path.join(protein_path, pair)
            conformer_file = os.path.join(pair_path, "aligned_to_start_with_hydrogen_conformers.mae")
            conformers = list(structure.StructureReader(conformer_file))[:args.num_conformers]
            grouped_conformers = group_files(args.conformer_n, conformers)

            grid_size = get_grid_size(pair_path, target, start)
            grouped_grid_locs = group_grid(args.grid_n, grid_size, 2)
            pose_path = os.path.join(pair_path, 'exhaustive_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size))

            for i in range(len(grouped_grid_locs)):
                for j in range(len(grouped_conformers)):
                    file1 = os.path.join(pose_path, 'exhaustive_search_info_{}_{}.csv'.format(i, j))
                    file2 = os.path.join(pose_path, 'exhaustive_search_poses_{}_{}.csv'.format(i, j))
                    counter += 1
                    if not os.path.exists(file1):
                        missing.append((protein, target, start, i, j))
                        continue
                    if not os.path.exists(file2):
                        missing.append((protein, target, start, i, j))

        print('Missing: {}/{}'.format(len(missing), counter))
        print(missing)

    elif args.task == 'delete':
        pairs = get_prots(args.docked_prot_file)
        random.shuffle(pairs)
        for protein, target, start in pairs[:5]:
            pair = '{}-to-{}'.format(target, start)
            protein_path = os.path.join(args.raw_root, protein)
            pair_path = os.path.join(protein_path, pair)
            pose_path = os.path.join(pair_path, 'exhaustive_grid_6_2_rotation_0_360_20_rmsd_2.5')
            os.system('rm -rf {}'.format(pose_path))

if __name__ == "__main__":
    main()
