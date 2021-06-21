"""
The purpose of this code is to create conformers

It can be run on sherlock using
$ $SCHRODINGER/run python3 search.py group /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits/search_test_incorrect_glide_index.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw --protein P03368 --target 1gno --start 1zp8 --grid_index 0 --conformer_index 97
"""

import argparse
import os
import schrodinger.structure as structure
from schrodinger.structutils.transform import get_centroid
import schrodinger.structutils.rmsd as rmsd
import random
import pickle
import scipy.spatial
import time
import math
import numpy as np
import pandas as pd

X_AXIS = [1.0, 0.0, 0.0]  # x-axis unit vector
Y_AXIS = [0.0, 1.0, 0.0]  # y-axis unit vector
Z_AXIS = [0.0, 0.0, 1.0]  # z-axis unit vector

# HELPER FUNCTIONS


def get_prots(docked_prot_file):
    """
    gets list of all protein, target ligands, and starting ligands in the index file
    :param docked_prot_file: (string) file listing proteins to process
    :return: process (list) list of all protein, target ligands, and starting ligands to process
    """
    process = []
    with open(docked_prot_file) as fp:
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


def get_conformer_prots(process, raw_root, num_pairs):
    conformer_prots = []
    for protein, target, start in process:
        if len(conformer_prots) == num_pairs:
            break
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, pair)

        if os.path.exists(os.path.join(pair_path, 'aligned_to_start_without_hydrogen_conformers.mae')):
            conformer_prots.append((protein, target, start))

    return conformer_prots


def get_grid_prots(grid_file, grid_size):
    infile = open(grid_file, 'rb')
    data = pickle.load(infile)
    infile.close()

    grid_prots = []
    for pair in data:
        if sum(data[pair][:grid_size + 1]) != 0:
            grid_prots.append(pair)

    return grid_prots


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


def modify_file(path, name):
    reading_file = open(path, "r")
    file_name = path.split('/')[-1]

    new_file_content = ""
    for line in reading_file:
        if line.strip() == name:
            new_line = line.replace(name, file_name)
        else:
            new_line = line
        new_file_content += new_line
    reading_file.close()

    writing_file = open(path, "w")
    writing_file.write(new_file_content)
    writing_file.close()


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
    at = at * 2
    at = (np.around(at - 0.5)).astype(np.int16)
    origin = np.full((3), np.amin(at))
    at = at - origin
    dim = np.amax(at) + 1
    grid = np.zeros((dim, dim, dim))
    np.add.at(grid, (at[:, 0], at[:, 1], at[:, 2]), 1)

    return grid, origin


def get_clash(s, grid, origin):
    at = s.getXYZ(copy=True)
    at = at * 2
    at = (np.around(at - 0.5)).astype(np.int16)
    at = at - origin
    return np.sum(grid[at[:, 0], at[:, 1], at[:, 2]])

# MAIN TASK FUNCTIONS


def all_search(grouped_files, mode, raw_root, run_path, docked_prot_file, rotation_search_step_size, grid_size, n, group_name,
               include_clash_filter, pocket_only, no_prot_h, num_conformers, protein, target, start):
    for i in range(len(grouped_files)):
        cmd = 'sbatch -p rondror -t 6:00:00 -o {} --wrap="$SCHRODINGER/run python3 search.py group {} {} {} {} ' \
              '--rotation_search_step_size {} --grid_size {} --grid_n {} --num_conformers {} --index {} --group_name {} ' \
              '--protein {} --target {} --start {}'
        if not include_clash_filter:
            cmd += ' --no_clash_filter'
        if not pocket_only:
            cmd += ' --all_prot'
        if not no_prot_h:
            cmd += ' --keep_prot_h'
        cmd += '"'
        out_file_name = 'search_{}_{}_{}_{}.out'.format(protein, target, start, i)
        os.system(cmd.format(os.path.join(run_path, out_file_name), mode, docked_prot_file, run_path, raw_root,
                             rotation_search_step_size, grid_size, n, num_conformers, i, group_name, protein, target,
                             start))


def check_pose(num_poses_searched, i, grid_loc, start_prot_grid, start_origin, c_indices, target_lig,
               target_lig_indices, rmsd_cutoff, num_correct, num_after_simple_filter, num_correct_after_simple_filter,
               target_prot_grid, target_origin, saved_dict, c, x, y, z):
    # look at potential pose
    num_poses_searched += 1

    # check simple filter
    start_clash = get_clash(c, start_prot_grid, start_origin)

    # check if pose correct
    rmsd_val = rmsd.calculate_in_place_rmsd(c, c_indices, target_lig, target_lig_indices)
    if rmsd_val < rmsd_cutoff:
        num_correct += 1
        name = '{}_{},{},{}_{},{},{}'.format(i, grid_loc[0], grid_loc[1], grid_loc[2], x, y,
                                             z)
        print(name, start_clash, rmsd_val)

    if start_clash == 0:
        num_after_simple_filter += 1
        if rmsd_val < rmsd_cutoff:
            num_correct_after_simple_filter += 1
        # save info for pose
        target_clash = get_clash(c, target_prot_grid, target_origin)
        name = '{}_{},{},{}_{},{},{}'.format(i, grid_loc[0], grid_loc[1], grid_loc[2], x, y,
                                             z)
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

    if rmsd_val < rmsd_cutoff:
        print(num_after_simple_filter, num_correct_after_simple_filter)

    return num_poses_searched, num_correct, num_after_simple_filter, num_correct_after_simple_filter


def rotate_pose(min_angle, max_angle, rotation_search_step_size, coords, from_origin_matrix, to_origin_matrix, c,
                num_poses_searched, i, grid_loc, start_prot_grid, start_origin, c_indices, target_lig,
                target_lig_indices, rmsd_cutoff, num_correct, num_after_simple_filter, num_correct_after_simple_filter,
                target_prot_grid, target_origin, saved_dict):
    for x in range(min_angle, max_angle + rotation_search_step_size, rotation_search_step_size):
        # rotation preprocessing
        rot_matrix_x = get_rotation_matrix(X_AXIS, x)

        for y in range(min_angle, max_angle + rotation_search_step_size, rotation_search_step_size):
            # rotation preprocessing
            rot_matrix_y = get_rotation_matrix(Y_AXIS, y)

            for z in range(min_angle, max_angle + rotation_search_step_size, rotation_search_step_size):
                # rotation preprocessing
                rot_matrix_z = get_rotation_matrix(Z_AXIS, z)

                # apply x,y,z rotation
                new_coords = rotate_structure(coords, from_origin_matrix, to_origin_matrix, rot_matrix_x,
                                              rot_matrix_y, rot_matrix_z)
                c.setXYZ(new_coords)

                num_poses_searched, num_correct, num_after_simple_filter, num_correct_after_simple_filter = \
                    check_pose(num_poses_searched, i, grid_loc, start_prot_grid, start_origin, c_indices, target_lig,
                               target_lig_indices, rmsd_cutoff, num_correct, num_after_simple_filter,
                               num_correct_after_simple_filter, target_prot_grid, target_origin, saved_dict, c, x, y, z)

                c.setXYZ(coords)

    return num_poses_searched, num_correct, num_after_simple_filter, num_correct_after_simple_filter


def search(protein, target, start, raw_root, rmsd_cutoff, start_clash_cutoff, rotation_search_step_size, num_conformers,
           grid_n, grid_search_step_size, grid_index, min_angle, max_angle, conformer_n, conformer_index):
    # important dirs
    pair = '{}-to-{}'.format(target, start)
    protein_path = os.path.join(raw_root, protein)
    pair_path = os.path.join(protein_path, pair)

    start_lig_file = os.path.join(pair_path, '{}_lig.mae'.format(start))
    start_lig = list(structure.StructureReader(start_lig_file))[0]
    start_center = get_centroid(start_lig)

    target_lig_file = os.path.join(pair_path, 'ligand_poses', '{}_lig0.mae'.format(target))
    target_lig = list(structure.StructureReader(target_lig_file))[0]
    target_center = get_centroid(target_lig)
    target_x, target_y, target_z, _ = target_center
    # get non hydrogen atom indices for rmsd
    target_lig_indices = [a.index for a in target_lig.atom if a.element != 'H']

    start_prot_file = os.path.join(pair_path, '{}_prot.mae'.format(start))
    start_prot = list(structure.StructureReader(start_prot_file))[0]

    target_prot_file = os.path.join(pair_path, '{}_prot.mae'.format(target))
    target_prot = list(structure.StructureReader(target_prot_file))[0]

    # get conformers
    conformer_file = os.path.join(pair_path, "aligned_to_start_with_hydrogen_conformers.mae")
    conformers = list(structure.StructureReader(conformer_file))[:num_conformers]
    conformer_indices = [i for i in range(len(conformers))]
    grouped_conformer_indices = group_files(conformer_n, conformer_indices)
    conformer_group_indices = grouped_conformer_indices[conformer_index]

    # get grid
    dist = np.sqrt((target_center[0] - start_center[0]) ** 2 +
                   (target_center[1] - start_center[1]) ** 2 +
                   (target_center[2] - start_center[2]) ** 2)

    grid_size = int(dist + 1)
    if grid_size % 2 == 1:
        grid_size += 1
    grid_size = 1
    grid_search_step_size = 1
    grouped_files = group_grid(grid_n, grid_size, grid_search_step_size)
    grid = grouped_files[grid_index]

    # get save location
    group_name = 'exhaustive_target_center_grid_{}_{}_rotation_{}_{}_{}_rmsd_{}'.format(grid_size,
                 grid_search_step_size, min_angle, max_angle, rotation_search_step_size, rmsd_cutoff)
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

    with open(os.path.join(pose_path, 'exhaustive_search_poses_{}_{}.csv'.format(grid_index, conformer_index)),
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
            # translate_structure(c, grid_loc[0], grid_loc[1], grid_loc[2])
            conformer_x, conformer_y, conformer_z, _ = list(get_centroid(c))
            translate_structure(c, -conformer_x + target_x + grid_loc[0], -conformer_y + target_y + grid_loc[1],
                                -conformer_z + target_z + grid_loc[2])
            conformer_center = list(get_centroid(c))
            coords = c.getXYZ(copy=True)

            # rotation preprocessing
            displacement_vector = get_coords_array_from_list(conformer_center)
            to_origin_matrix = get_translation_matrix(-1 * displacement_vector)
            from_origin_matrix = get_translation_matrix(displacement_vector)

            num_poses_searched, num_correct, num_after_simple_filter, num_correct_after_simple_filter = \
                rotate_pose(min_angle, max_angle, rotation_search_step_size, coords, from_origin_matrix,
                            to_origin_matrix, c, num_poses_searched, i, grid_loc, start_prot_grid, start_origin,
                            c_indices, target_lig, target_lig_indices, rmsd_cutoff, num_correct,
                            num_after_simple_filter, num_correct_after_simple_filter, target_prot_grid, target_origin,
                            saved_dict)

            translate_structure(c, -grid_loc[0], -grid_loc[1], -grid_loc[2])

        with open(os.path.join(pose_path, 'exhaustive_search_poses_{}_{}.csv'.format(grid_index, conformer_index)),
                  'a') as f:
            df = pd.DataFrame.from_dict(saved_dict)
            df.to_csv(f, header=False)

    # save info for grid_loc
    decoy_end_time = time.time()
    data_dict['protein'].append(protein)
    data_dict['target'].append(target)
    data_dict['start'].append(start)
    data_dict['num_conformers'].append(len(conformer_group_indices))
    data_dict['num_grid_locs'].append(len(grid))
    data_dict['num_poses_searched'].append(num_poses_searched)
    data_dict['num_correct'].append(num_correct)
    data_dict['num_after_simple_filter'].append(num_after_simple_filter)
    data_dict['num_correct_after_simple_filter'].append(num_correct_after_simple_filter)
    data_dict['time_elapsed'].append(decoy_end_time - decoy_start_time)

    if grid_index == -1:
        df = pd.DataFrame.from_dict(data_dict)
        df.to_csv(os.path.join(pair_path, 'exhaustive_search_info.csv'))
    else:
        df = pd.DataFrame.from_dict(data_dict)
        df.to_csv(os.path.join(pose_path, 'exhaustive_search_info_{}_{}.csv'.format(grid_index, conformer_index)))


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
    # parser.add_argument('--grid_n', type=int, default=60, help='number of grid_points processed in each job')
    parser.add_argument('--grid_n', type=int, default=40, help='number of grid_points processed in each job')
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
    # parser.add_argument('--conformer_n', type=int, default=2, help='number of grid_points processed in each job')
    parser.add_argument('--conformer_n', type=int, default=1, help='number of grid_points processed in each job')
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
        pairs = get_prots(args.docked_prot_file)
        random.shuffle(pairs)
        counter = 0
        protein = 'P03368'
        target = '1gno'
        start = '1zp8'
        # for protein, target, start in pairs[:5]:
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(args.raw_root, protein)
        pair_path = os.path.join(protein_path, pair)
        conformer_file = os.path.join(pair_path, "aligned_to_start_with_hydrogen_conformers.mae")
        conformers = list(structure.StructureReader(conformer_file))[:args.num_conformers]
        grouped_conformers = group_files(args.conformer_n, conformers)

        target_lig_file = os.path.join(pair_path, 'ligand_poses', '{}_lig0.mae'.format(target))
        target_lig = list(structure.StructureReader(target_lig_file))[0]
        target_center = get_centroid(target_lig)

        start_lig_file = os.path.join(pair_path, '{}_lig.mae'.format(start))
        start_lig = list(structure.StructureReader(start_lig_file))[0]
        start_center = get_centroid(start_lig)

        dist = np.sqrt((target_center[0] - start_center[0]) ** 2 +
                       (target_center[1] - start_center[1]) ** 2 +
                       (target_center[2] - start_center[2]) ** 2)

        # grid_size = int(dist + 1)
        # if grid_size % 2 == 1:
        #     grid_size += 1
        # grouped_grid_locs = group_grid(args.grid_n, grid_size, 2)

        grid_size = 1
        grid_search_step_size = 1
        grouped_grid_locs = group_grid(args.grid_n, grid_size, grid_search_step_size)

        print('protein: {}, target: {}, start: {}, num_conformers: {}'.format(protein, target, start, len(conformers)))

        for i in range(len(grouped_grid_locs)):
            for j in range(len(grouped_conformers)):
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
        search(args.protein, args.target, args.start, args.raw_root, args.rmsd_cutoff, args.start_clash_cutoff,
               args.rotation_search_step_size, args.num_conformers, args.grid_n, args.grid_search_step_size,
               args.grid_index, args.min_angle, args.max_angle, args.conformer_n, args.conformer_index)

    elif args.task == 'check':
        if args.mode == 'train':
            pairs = get_prots(args.docked_prot_file)
            check_search(pairs, args.raw_root)
        elif args.mode == 'test' and args.protein == '':
            pairs = get_prots(args.docked_prot_file)
            random.shuffle(pairs)
            grouped_files = group_grid(args.grid_n, args.grid_size)
            unfinished = []
            for protein, target, start in pairs[:5]:
                pair = '{}-to-{}'.format(target, start)
                protein_path = os.path.join(args.raw_root, protein)
                pair_path = os.path.join(protein_path, pair)
                pose_path = os.path.join(pair_path, args.group_name)
                for i in range(len(grouped_files)):
                    if not os.path.exists(os.path.join(pose_path, 'exhaustive_search_poses_{}.csv'.format(i))):
                        unfinished.append((protein, target, start))
                        break

            print('Missing {} / 5'.format(len(unfinished)))
            print(unfinished)


if __name__ == "__main__":
    main()
