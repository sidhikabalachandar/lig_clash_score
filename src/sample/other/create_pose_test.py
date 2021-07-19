"""
The purpose of this code is to create conformers

It can be run on sherlock using
$ $SCHRODINGER/run python3 create_pose_test.py /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw --protein P11838 --target 3wz6 --start 1gvx --conformer_index 28 --grid_loc_x 2 --grid_loc_y 4 --grid_loc_z 0 --rot_x 80 --rot_y 40 --rot_z 320
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('--num_conformers', type=int, default=300, help='maximum number of conformers considered')
    parser.add_argument('--protein', type=str, default='', help='name of protein')
    parser.add_argument('--target', type=str, default='', help='name of target ligand')
    parser.add_argument('--start', type=str, default='', help='name of start ligand')
    parser.add_argument('--conformer_index', type=int, default=97, help='conformer_index')
    parser.add_argument('--grid_loc_x', type=int, default=0, help='grid_loc_x')
    parser.add_argument('--grid_loc_y', type=int, default=0, help='grid_loc_y')
    parser.add_argument('--grid_loc_z', type=int, default=0, help='grid_loc_z')
    parser.add_argument('--rot_x', type=int, default=180, help='rot_x')
    parser.add_argument('--rot_y', type=int, default=320, help='rot_y')
    parser.add_argument('--rot_z', type=int, default=180, help='rot_z')
    args = parser.parse_args()

    # important dirs
    pair = '{}-to-{}'.format(args.target, args.start)
    protein_path = os.path.join(args.raw_root, args.protein)
    pair_path = os.path.join(protein_path, pair)

    target_lig_file = os.path.join(pair_path, 'ligand_poses', '{}_lig0.mae'.format(args.target))
    target_lig = list(structure.StructureReader(target_lig_file))[0]
    target_lig_indices = [a.index for a in target_lig.atom if a.element != 'H']

    # get conformers
    conformer_file = os.path.join(pair_path, "aligned_to_start_with_hydrogen_conformers.mae")
    conformers = list(structure.StructureReader(conformer_file))[:args.num_conformers]

    c = conformers[args.conformer_index]
    c_indices = [a.index for a in c.atom if a.element != 'H']

    translate_structure(c, args.grid_loc_x, args.grid_loc_y, args.grid_loc_z)
    conformer_center = list(get_centroid(c))
    coords = c.getXYZ(copy=True)

    # rotation preprocessing
    displacement_vector = get_coords_array_from_list(conformer_center)
    to_origin_matrix = get_translation_matrix(-1 * displacement_vector)
    from_origin_matrix = get_translation_matrix(displacement_vector)

    rot_matrix_x = get_rotation_matrix(X_AXIS, math.radians(args.rot_x))
    rot_matrix_y = get_rotation_matrix(Y_AXIS, math.radians(args.rot_y))
    rot_matrix_z = get_rotation_matrix(Z_AXIS, math.radians(args.rot_z))

    # apply x,y,z rotation
    new_coords = rotate_structure(coords, from_origin_matrix, to_origin_matrix, rot_matrix_x,
                                  rot_matrix_y, rot_matrix_z)
    c.setXYZ(new_coords)

    # check if pose correct
    rmsd_val = rmsd.calculate_in_place_rmsd(c, c_indices, target_lig, target_lig_indices)
    print(rmsd_val)
    rmsd_val = rmsd.calculate_in_place_rmsd(c, c.getAtomIndices(), target_lig, target_lig.getAtomIndices())
    print(rmsd_val)

    with structure.StructureWriter('pose.mae') as pose:
        pose.append(c)


if __name__ == "__main__":
    main()
