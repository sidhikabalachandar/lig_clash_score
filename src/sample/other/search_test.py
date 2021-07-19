"""
The purpose of this code is to create conformers

It can be run on sherlock using
$ $SCHRODINGER/run python3 search.py all test /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits/search_test_incorrect_glide_index.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw --group_name exhaustive_grid_1_rotation_0_360_10 --protein P02829 --target 2fxs --start 2weq --index 0
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


def group_grid(n, grid_size):
    grid = []
    for dx in range(-grid_size, grid_size + 1):
        for dy in range(-grid_size, grid_size + 1):
            for dz in range(-grid_size, grid_size + 1):
                grid.append([dx, dy, dz])

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


def rotate_structure(coords, x_angle, y_angle, z_angle, rot_center):
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

    # This action is achieved in four steps
    # 1)  Find the vector that moves the rot_center to the origin
    # 2)  Move the structure along that vector
    # 3)  Apply rotations
    # 4)  Move the structure back

    displacement_vector = get_coords_array_from_list(rot_center)
    to_origin_matrix = get_translation_matrix(-1 * displacement_vector)
    rot_matrix_x = get_rotation_matrix(X_AXIS, x_angle)
    rot_matrix_y = get_rotation_matrix(Y_AXIS, y_angle)
    rot_matrix_z = get_rotation_matrix(Z_AXIS, z_angle)
    from_origin_matrix = get_translation_matrix(displacement_vector)

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


def search(protein, target, start, raw_root, get_time, rmsd_cutoff, start_clash_cutoff, rotation_search_step_size,
           no_prot_h, pocket_only, num_conformers, grid, grid_index, group_name, min_angle, max_angle, conformer_n,
           conformer_index, test=False,  x_rot=0, y_rot=0, z_rot=0):
    pair = '{}-to-{}'.format(target, start)
    protein_path = os.path.join(raw_root, protein)
    pair_path = os.path.join(protein_path, pair)
    target_lig_file = os.path.join(pair_path, 'ligand_poses', '{}_lig0.mae'.format(target))
    target_lig = list(structure.StructureReader(target_lig_file))[0]
    build.delete_hydrogens(target_lig)
    start_prot_file = os.path.join(pair_path, '{}_prot.mae'.format(start))
    start_prot = list(structure.StructureReader(start_prot_file))[0]
    target_prot_file = os.path.join(pair_path, '{}_prot.mae'.format(target))
    target_prot = list(structure.StructureReader(target_prot_file))[0]
    conformer_file = os.path.join(pair_path, "aligned_to_start_without_hydrogen_conformers.mae")
    conformers = list(structure.StructureReader(conformer_file))[:num_conformers]
    grouped_conformers = group_files(conformer_n, conformers)
    conformer_group = grouped_conformers[conformer_index]

    if pocket_only:
        get_pocket_res(start_prot, target_lig, 6)
        get_pocket_res(target_prot, target_lig, 6)
    if no_prot_h:
        build.delete_hydrogens(start_prot)
        build.delete_hydrogens(target_prot)

    if get_time:
        get_times(pair_path, target_lig, start_prot)
    else:
        conformer = create_poses(grid, target_lig, start_prot, target_prot, rmsd_cutoff, start_clash_cutoff,
                                 rotation_search_step_size, protein, target, start, pair_path, test, x_rot, y_rot,
                                 z_rot, conformer_group, grid_index, conformer_index, group_name, min_angle, max_angle)

        return conformer


def get_times(pair_path, target_lig, prot):
    schrodinger_translate_times = []
    schrodinger_rotate_times = []
    custom_translate_times = []
    custom_rotate_times = []
    clash_iterator_times = []
    clash_volume_times = []
    rmsd_times = []
    conformer_file = os.path.join(pair_path, "aligned_to_start_without_hydrogen_conformers.mae")
    conformers = list(structure.StructureReader(conformer_file))

    for conformer in conformers:
        # schrdoinger translation
        grid_loc = [0, 0, 0]
        start = time.time()
        transform.translate_structure(conformer, grid_loc[0], grid_loc[1], grid_loc[2])
        end = time.time()
        schrodinger_translate_times.append(end - start)

        # schrdoinger rotation
        conformer_center = list(get_centroid(conformer))
        start = time.time()
        transform.rotate_structure(conformer, math.radians(-30), math.radians(-30), math.radians(-30), conformer_center)
        end = time.time()
        schrodinger_rotate_times.append(end - start)

        # custom translation
        grid_loc = [0, 0, 0]
        start = time.time()
        translate_structure(conformer, grid_loc[0], grid_loc[1], grid_loc[2])
        end = time.time()
        custom_translate_times.append(end - start)

        # custom rotation
        conformer_center = list(get_centroid(conformer))
        start = time.time()
        rotate_structure(conformer, math.radians(-30), math.radians(-30), math.radians(-30), conformer_center)
        end = time.time()
        custom_rotate_times.append(end - start)

        # get clash_iterator
        start = time.time()
        max([x[2] for x in list(steric_clash.clash_iterator(prot, struc2=conformer))])
        end = time.time()
        clash_iterator_times.append(end - start)

        # get clash_volume
        start = time.time()
        steric_clash.clash_volume(prot, struc2=conformer)
        end = time.time()
        clash_volume_times.append(end - start)

        # get rmsd
        start = time.time()
        rmsd.calculate_in_place_rmsd(conformer, conformer.getAtomIndices(), target_lig,
                                     target_lig.getAtomIndices())
        end = time.time()
        rmsd_times.append(end - start)

    print("Average schrodinger translate time =", statistics.mean(schrodinger_translate_times))
    print("Average schrodinger rotate time =", statistics.mean(schrodinger_rotate_times))
    print("Average custom translate time =", statistics.mean(custom_translate_times))
    print("Average custom rotate time =", statistics.mean(custom_rotate_times))
    print("Average clash iterator time =", statistics.mean(clash_iterator_times))
    print("Average clash volume time =", statistics.mean(clash_volume_times))
    print("Average rmsd time =", statistics.mean(rmsd_times))

def check_clash_vol(other_c, other_prot):
    protein = 'O38732'
    target = '2i0a'
    start = '2q5k'
    raw_root = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw'
    group_name = 'exhaustive_grid_1_rotation_5'
    n = 2000
    file_index = 0
    name_index = 0
    pair = '{}-to-{}'.format(target, start)
    protein_path = os.path.join(raw_root, protein)
    pair_path = os.path.join(protein_path, pair)
    pose_path = os.path.join(pair_path, group_name)
    clash_path = os.path.join(pose_path, 'ideal_clash_data')
    if not os.path.exists(clash_path):
        os.mkdir(clash_path)
    file = os.path.join(pose_path, 'exhaustive_search_poses_{}.csv'.format(file_index))
    df = pd.read_csv(file)
    grouped_files = group_files(n, df['name'].to_list())
    names = grouped_files[name_index]
    prot_target = list(structure.StructureReader(os.path.join(pair_path, '{}_prot.mae'.format(target))))[0]
    conformer_file = os.path.join(pair_path, "aligned_to_start_with_hydrogen_conformers.mae")
    conformers = list(structure.StructureReader(conformer_file))

    for name in names:
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
        new_coords = rotate_structure(coords, math.radians(rot_x), math.radians(rot_y), math.radians(rot_z),
                                      conformer_center)
        c.setXYZ(new_coords)
        if name == '0_-1,-1,-1_-20,-20,10':
            print('target clash volume in function 2:', steric_clash.clash_volume(prot_target, struc2=c))
            rmsd_val = rmsd.calculate_in_place_rmsd(c, c.getAtomIndices(), other_c, other_c.getAtomIndices())
            print('c rmsd:', rmsd_val)
            rmsd_val = rmsd.calculate_in_place_rmsd(prot_target, prot_target.getAtomIndices(), other_prot,
                                                    other_prot.getAtomIndices())
            print('prot rmsd:', rmsd_val)
            return
        c.setXYZ(old_coords)

def create_poses(grid, target_lig, start_prot, target_prot, rmsd_cutoff, start_clash_cutoff, rotation_search_step_size,
                 protein, target, start, pair_path, test, x_rot, y_rot, z_rot, conformers, grid_index, conformer_index,
                 group_name, min_angle, max_angle):
    data_dict = {'protein': [], 'target': [], 'start': [], 'num_conformers': [], 'num_poses_searched': [],
                 'num_correct_poses_found': [], 'num_correct_after_simple_filter': [], 'time_elapsed': [],
                 'time_elapsed_per_conformer': [], 'grid_loc_x': [], 'grid_loc_y': [], 'grid_loc_z': []}
    saved_dict = {'name': [], 'conformer_index': [], 'grid_loc_x': [], 'grid_loc_y': [], 'grid_loc_z': [], 'rot_x': [],
                  'rot_y': [], 'rot_z': [], 'start_clash': [], 'target_clash': [], 'rmsd': []}

    for grid_loc in grid:
        counter = 0
        num_correct_found = 0
        num_correct_after_simple_filter = 0
        decoy_start_time = time.time()
        for i, c in enumerate(conformers):
            translate_structure(c, grid_loc[0], grid_loc[1], grid_loc[2])
            conformer_center = list(get_centroid(c))
            coords = c.getXYZ(copy=True)

            for x in range(min_angle, max_angle + rotation_search_step_size, rotation_search_step_size):
                for y in range(min_angle, max_angle + rotation_search_step_size, rotation_search_step_size):
                    for z in range(min_angle, max_angle + rotation_search_step_size, rotation_search_step_size):
                        counter += 1
                        new_coords = rotate_structure(coords, math.radians(x), math.radians(y), math.radians(z),
                                                      conformer_center)
                        c.setXYZ(new_coords)

                        if test and x_rot == x and y_rot == y and z_rot == z:
                            return c

                        rmsd_val = rmsd.calculate_in_place_rmsd(c, c.getAtomIndices(), target_lig,
                                                                target_lig.getAtomIndices())
                        if rmsd_val < rmsd_cutoff:
                            num_correct_found += 1

                        start_clash = steric_clash.clash_volume(start_prot, struc2=c)

                        if start_clash < start_clash_cutoff:
                            if rmsd_val < rmsd_cutoff:
                                num_correct_after_simple_filter += 1
                            target_clash = steric_clash.clash_volume(target_prot, struc2=c)
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

        decoy_end_time = time.time()
        data_dict['protein'].append(protein)
        data_dict['target'].append(target)
        data_dict['start'].append(start)
        data_dict['num_conformers'].append(len(conformers))
        data_dict['num_poses_searched'].append(counter)
        data_dict['num_correct_poses_found'].append(num_correct_found)
        data_dict['num_correct_after_simple_filter'].append(num_correct_after_simple_filter)
        data_dict['time_elapsed'].append(decoy_end_time - decoy_start_time)
        data_dict['time_elapsed_per_conformer'].append((decoy_end_time - decoy_start_time) / len(conformers))
        data_dict['grid_loc_x'].append(grid_loc[0])
        data_dict['grid_loc_y'].append(grid_loc[1])
        data_dict['grid_loc_z'].append(grid_loc[2])

    if grid_index == -1:
        df = pd.DataFrame.from_dict(saved_dict)
        df.to_csv(os.path.join(pair_path, 'exhaustive_search_poses.csv'))

        df = pd.DataFrame.from_dict(data_dict)
        df.to_csv(os.path.join(pair_path, 'exhaustive_search_info.csv'))
    else:
        pose_path = os.path.join(pair_path, group_name)
        if not os.path.exists(pose_path):
            os.mkdir(pose_path)
        df = pd.DataFrame.from_dict(saved_dict)
        df.to_csv(os.path.join(pose_path, 'exhaustive_search_poses_{}_{}.csv'.format(grid_index, conformer_index)))

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


def test_rotate_translate(protein, target, start, raw_root):
    pair = '{}-to-{}'.format(target, start)
    protein_path = os.path.join(raw_root, protein)
    pair_path = os.path.join(protein_path, pair)
    prot_file = os.path.join(pair_path, '{}_prot.mae'.format(start))
    schrodinger_prot = list(structure.StructureReader(prot_file))[0]
    custom_prot = list(structure.StructureReader(prot_file))[0]
    translation_vector = np.random.uniform(low=-100, high=100, size=(3))
    transform.translate_structure(schrodinger_prot, translation_vector[0], translation_vector[1],
                                  translation_vector[2])
    translate_structure(custom_prot, translation_vector[0], translation_vector[1],
                        translation_vector[2])
    schrodinger_atoms = np.array(schrodinger_prot.getXYZ(copy=False))
    custom_atoms = np.array(custom_prot.getXYZ(copy=False))
    if np.array_equal(schrodinger_atoms, custom_atoms):
        print("Translate function works properly")
    else:
        print("Error in translate function")

    schrodinger_prot = list(structure.StructureReader(prot_file))[0]
    custom_prot = list(structure.StructureReader(prot_file))[0]
    rotation_vector = np.random.uniform(low=-2 * np.pi, high=2 * np.pi, size=(3))
    rotation_center = np.random.uniform(low=-100, high=100, size=(3))
    rotation_center = [rotation_center[0], rotation_center[1], rotation_center[2]]
    transform.rotate_structure(schrodinger_prot, rotation_vector[0], rotation_vector[1], rotation_vector[2],
                               rotation_center)
    coords = rotate_structure(custom_prot.getXYZ(copy=False), rotation_vector[0], rotation_vector[1],
                              rotation_vector[2], rotation_center)
    custom_prot.setXYZ(coords)
    schrodinger_atoms = np.array(schrodinger_prot.getXYZ(copy=False))
    custom_atoms = np.array(custom_prot.getXYZ(copy=False))
    if np.amax(np.absolute(schrodinger_atoms - custom_atoms)) < 10 ** -7:
        print("Rotate function works properly")
    else:
        print("Error in rotate function")


def test_search(protein, target, start, raw_root, save_folder, cutoff, rotation_search_step_size, no_prot_h,
                pocket_only, get_time, include_clash_filter, inc_target_clash_cutoff):
    angles = [i for i in range(-30, 30 + rotation_search_step_size, rotation_search_step_size)]
    angles = angles[:5]
    x_rot = random.choice(angles)
    y_rot = random.choice(angles)
    z_rot = random.choice(angles)
    grid_points = [i for i in range(-6, 7)]
    grid = [[random.choice(grid_points), random.choice(grid_points), random.choice(grid_points)]]

    conformer = search(protein, target, start, 0, raw_root, save_folder, get_time, cutoff, rotation_search_step_size,
                       grid, no_prot_h, pocket_only, include_clash_filter, inc_target_clash_cutoff, True, x_rot, y_rot,
                       z_rot)

    pair = '{}-to-{}'.format(target, start)
    protein_path = os.path.join(raw_root, protein)
    pair_path = os.path.join(protein_path, pair)
    conformer_file = os.path.join(pair_path, "aligned_to_start_without_hydrogen_conformers.mae")
    base_conf = list(structure.StructureReader(conformer_file))[0]
    translate_structure(base_conf, grid[0][0], grid[0][1], grid[0][2])
    base_conf_center = list(get_centroid(base_conf))
    coords = base_conf.getXYZ(copy=False)
    new_coords = rotate_structure(coords, math.radians(x_rot), math.radians(y_rot), math.radians(z_rot),
                                  base_conf_center)
    base_conf.setXYZ(new_coords)

    rmsd_val = rmsd.calculate_in_place_rmsd(conformer, conformer.getAtomIndices(), base_conf,
                                            base_conf.getAtomIndices())
    if abs(rmsd_val) == 0:
        print("Search works properly", rmsd_val)
    else:
        print("x_rot =", x_rot, "y_rot =", y_rot, "z_rot =", z_rot)
        print("RMSD =", rmsd_val, "but RMSD should equal 0")



def get_grid(df, center, config, rot_mat=np.eye(3, 3)):
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
    size = grid_size(config)
    true_radius = size * config.resolution / 2.0

    # Select valid atoms.
    at = df[['x', 'y', 'z']].values.astype(np.float32)
    elements = df['element'].values

    # Center atoms.
    at = at - center

    # Apply rotation matrix.
    at = np.dot(at, rot_mat)
    at = (np.around((at + true_radius) / config.resolution - 0.5)).astype(np.int16)

    # Prune out atoms outside of grid as well as non-existent atoms.
    sel = np.all(at >= 0, axis=1) & np.all(at < size, axis=1) & (elements != '')
    at = at[sel]

    # Form final grid.
    labels = elements[sel]
    lsel = np.nonzero([_recognized(x, config.element_mapping) for x in labels])
    labels = labels[lsel]
    labels = np.array([config.element_mapping[x] for x in labels], dtype=np.int8)

    grid = np.zeros(grid_shape(config), dtype=np.float32)
    grid[at[lsel, 0], at[lsel, 1], at[lsel, 2], labels] = 1

    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either align or search')
    parser.add_argument('mode', type=str, help='either train or test')
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('--num_conformers', type=int, default=300, help='maximum number of conformers considered')
    parser.add_argument('--num_pairs', type=int, default=10, help='number of protein-ligand pairs considered')
    parser.add_argument('--grid_n', type=int, default=1, help='number of grid_points processed in each job')
    parser.add_argument('--grid_size', type=int, default=1, help='grid size in positive and negative x, y, z '
                                                                 'directions')
    parser.add_argument('--grid_file', type=str, default='', help='pickle file with grid data dictionary')
    parser.add_argument('--grid_index', type=int, default=-1, help='grid point group index')
    parser.add_argument('--rotation_search_step_size', type=int, default=10, help='step size between each angle '
                                                                                 'checked, in degrees')
    parser.add_argument('--min_angle', type=int, default=0, help='min angle of rotation in degrees')
    parser.add_argument('--max_angle', type=int, default=360, help='min angle of rotation in degrees')
    parser.add_argument('--conformer_n', type=int, default=10, help='number of grid_points processed in each job')
    parser.add_argument('--conformer_index', type=int, default=-1, help='number of grid_points processed in each job')
    parser.add_argument('--rmsd_cutoff', type=int, default=2, help='rmsd accuracy cutoff between predicted ligand pose '
                                                                   'and true ligand pose')
    parser.add_argument('--start_clash_cutoff', type=int, default=100, help='clash cutoff between start protein and '
                                                                            'ligand pose')
    parser.add_argument('--group_name', type=str, default='', help='name of pose group subdir')
    parser.add_argument('--protein', type=str, default='', help='name of protein')
    parser.add_argument('--target', type=str, default='', help='name of target ligand')
    parser.add_argument('--start', type=str, default='', help='name of start ligand')
    parser.add_argument('--clash_filter', dest='include_clash_filter', action='store_true')
    parser.add_argument('--no_clash_filter', dest='include_clash_filter', action='store_false')
    parser.set_defaults(include_clash_filter=True)
    parser.add_argument('--prot_pocket_only', dest='pocket_only', action='store_true')
    parser.add_argument('--all_prot', dest='pocket_only', action='store_false')
    parser.set_defaults(pocket_only=True)
    parser.add_argument('--remove_prot_h', dest='no_prot_h', action='store_true')
    parser.add_argument('--keep_prot_h', dest='no_prot_h', action='store_false')
    parser.set_defaults(no_prot_h=True)
    parser.add_argument('--time', dest='get_time', action='store_true')
    parser.add_argument('--no_time', dest='get_time', action='store_false')
    parser.set_defaults(get_time=False)
    args = parser.parse_args()

    random.seed(0)

    if not os.path.exists(args.run_path):
        os.mkdir(args.run_path)

    if args.task == 'all':
        if args.mode == 'train':
            pairs = get_prots(args.docked_prot_file)
            grouped_files = group_files(args.grid_n, pairs)
            all_search(grouped_files, args.mode, args.raw_root, args.run_path, args.docked_prot_file,
                       args.rotation_search_step_size,
                       args.grid_size, args.grid_n, args.group_name, args.include_clash_filter, args.pocket_only,
                       args.no_prot_h,
                       args.num_conformers, args.protein, args.target, args.start)
        elif args.mode == 'test' and args.protein == '':
            pairs = get_prots(args.docked_prot_file)
            random.shuffle(pairs)
            grouped_files = group_grid(args.grid_n, args.grid_size)
            for protein, target, start in pairs[:5]:
                print(protein, target, start)
                all_search(grouped_files, args.mode, args.raw_root, args.run_path, args.docked_prot_file,
                           args.rotation_search_step_size,
                           args.grid_size, args.grid_n, args.group_name, args.include_clash_filter, args.pocket_only,
                           args.no_prot_h,
                           args.num_conformers, protein, target, start)
        elif args.mode == 'test':
            counter = 0
            grouped_files = group_grid(args.grid_n, args.grid_size)
            pair = '{}-to-{}'.format(args.target, args.start)
            protein_path = os.path.join(args.raw_root, args.protein)
            pair_path = os.path.join(protein_path, pair)
            conformer_file = os.path.join(pair_path, "aligned_to_start_without_hydrogen_conformers.mae")
            conformers = list(structure.StructureReader(conformer_file))[:args.num_conformers]
            grouped_conformers = group_files(args.conformer_n, conformers)
            for i in range(len(grouped_files)):
                for j in range(len(grouped_conformers)):
                    cmd = 'sbatch -p owners -t 6:00:00 -o {} --wrap="$SCHRODINGER/run python3 search.py group {} {} ' \
                          '{} {} --rotation_search_step_size {} --grid_size {} --grid_n {} --num_conformers {} ' \
                          '--conformer_n {} --grid_index {} --conformer_index {} --group_name {} --protein {} ' \
                          '--target {} --start {}'
                    if not args.include_clash_filter:
                        cmd += ' --no_clash_filter'
                    if not args.pocket_only:
                        cmd += ' --all_prot'
                    if not args.no_prot_h:
                        cmd += ' --keep_prot_h'
                    cmd += '"'
                    out_file_name = 'search_{}_{}_{}_{}_{}.out'.format(args.protein, args.target, args.start, i, j)
                    counter += 1
                    os.system(cmd.format(os.path.join(args.run_path, out_file_name), args.mode, args.docked_prot_file,
                                         args.run_path, args.raw_root, args.rotation_search_step_size, args.grid_size,
                                         args.grid_n, args.num_conformers, args.conformer_n, i, j, args.group_name,
                                         args.protein, args.target, args.start))

            print(counter)

    elif args.task == 'group':
        if args.protein == '':
            pairs = get_prots(args.docked_prot_file)
            grouped_files = group_files(args.grid_n, pairs)
            grid = []
            for dx in range(-args.grid_size, args.grid_size + 1):
                for dy in range(-args.grid_size, args.grid_size + 1):
                    for dz in range(-args.grid_size, args.grid_size + 1):
                        grid.append([dx, dy, dz])
            for protein, target, start in grouped_files[args.grid_index]:
                search(protein, target, start, args.raw_root, args.get_time, args.rmsd_cutoff, args.start_clash_cutoff,
                       args.rotation_search_step_size, args.no_prot_h, args.pocket_only, args.num_conformers, grid, -1,
                       args.group_name, args.min_angle, args.max_angle, args.conformer_n, args.conformer_index)
        else:
            grouped_files = group_grid(args.grid_n, args.grid_size)
            grid = grouped_files[args.grid_index]
            search(args.protein, args.target, args.start, args.raw_root, args.get_time, args.rmsd_cutoff,
                   args.start_clash_cutoff, args.rotation_search_step_size, args.no_prot_h, args.pocket_only,
                   args.num_conformers, grid, args.grid_index, args.group_name, args.min_angle, args.max_angle,
                   args.conformer_n, args.conformer_index)

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


    elif args.task == 'test_rotate_translate':
        test_rotate_translate(args.protein, args.target, args.start, args.raw_root)

    elif args.task == 'test_search':
        test_search(args.protein, args.target, args.start, args.raw_root, args.save_folder, args.cutoff,
                    args.rotation_search_step_size, args.no_prot_h, args.pocket_only, args.get_time,
                    args.include_clash_filter, args.inc_target_clash_cutoff)


if __name__ == "__main__":
    main()
