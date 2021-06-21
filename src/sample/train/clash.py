"""
The purpose of this code is to create conformers

It can be run on sherlock using
$ $SCHRODINGER/run python3 clash.py /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits/search_test_incorrect_glide_index.txt /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
"""

import argparse
import os
import schrodinger.structure as structure
from schrodinger.structutils.transform import get_centroid
import schrodinger.structutils.interactions.steric_clash as steric_clash
import schrodinger.structutils.rmsd as rmsd
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

X_AXIS = [1.0, 0.0, 0.0]  # x-axis unit vector
Y_AXIS = [0.0, 1.0, 0.0]  # y-axis unit vector
Z_AXIS = [0.0, 0.0, 1.0]  # z-axis unit vector


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
    parser = argparse.ArgumentParser()
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('raw_root', type=str, help='directory where script and output files will be written')
    args = parser.parse_args()
    random.seed(0)

    # if not os.path.exists('clash_graph.pkl'):
    pairs = get_prots(args.docked_prot_file)
    random.shuffle(pairs)
    correct_custom_clashes = []
    correct_schrod_clashes = []
    incorrect_custom_clashes = []
    incorrect_schrod_clashes = []

    for protein, target, start in pairs[:5]:
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(args.raw_root, protein)
        pair_path = os.path.join(protein_path, pair)
        start_prot_file = os.path.join(pair_path, '{}_prot.mae'.format(start))
        start_prot = list(structure.StructureReader(start_prot_file))[0]
        conformer_file = os.path.join(pair_path, "aligned_to_start_with_hydrogen_conformers.mae")
        conformers = list(structure.StructureReader(conformer_file))
        target_lig_file = os.path.join(pair_path, 'ligand_poses', '{}_lig0.mae'.format(target))
        target_lig = list(structure.StructureReader(target_lig_file))[0]

        prot_grid, origin = get_grid(start_prot)

        grid = []
        grid_size = 1
        for dx in range(-grid_size, grid_size + 1):
            for dy in range(-grid_size, grid_size + 1):
                for dz in range(-grid_size, grid_size + 1):
                    grid.append([dx, dy, dz])

        poses = []
        min_angle = 0
        max_angle = 360
        rotation_search_step_size = 10
        angles = [x for x in range(min_angle, max_angle + rotation_search_step_size, rotation_search_step_size)]
        for i in range(300):
            grid_loc = random.choice(grid)
            c = random.choice(conformers)
            x = random.choice(angles)
            y = random.choice(angles)
            z = random.choice(angles)
            poses.append((grid_loc, c, x, y, z))

        for grid_loc, c, x, y, z in poses:
            translate_structure(c, grid_loc[0], grid_loc[1], grid_loc[2])
            conformer_center = list(get_centroid(c))
            coords = c.getXYZ(copy=True)
            displacement_vector = get_coords_array_from_list(conformer_center)
            to_origin_matrix = get_translation_matrix(-1 * displacement_vector)
            from_origin_matrix = get_translation_matrix(displacement_vector)
            rot_matrix_x = get_rotation_matrix(X_AXIS, x)
            rot_matrix_y = get_rotation_matrix(Y_AXIS, y)
            rot_matrix_z = get_rotation_matrix(Z_AXIS, z)
            new_coords = rotate_structure(coords, from_origin_matrix, to_origin_matrix, rot_matrix_x,
                                          rot_matrix_y, rot_matrix_z)
            c.setXYZ(new_coords)
            custom_clash = get_clash(c, prot_grid, origin)
            schrod_clash = steric_clash.clash_volume(start_prot, struc2=c)

            # check if pose correct
            rmsd_val = rmsd.calculate_in_place_rmsd(c, c.getAtomIndices(), target_lig,
                                                    target_lig.getAtomIndices())
            if (rmsd_val < 2):
                correct_custom_clashes.append(custom_clash)
                correct_schrod_clashes.append(schrod_clash)
            else:
                incorrect_custom_clashes.append(custom_clash)
                incorrect_schrod_clashes.append(schrod_clash)

    outfile = open('clash_graph.pkl', 'wb')
    pickle.dump([correct_custom_clashes, correct_schrod_clashes, incorrect_custom_clashes, incorrect_schrod_clashes], outfile)
    # else:
    #     infile = open('clash_graph.pkl', 'rb')
    #     schrod_clashes, custom_clashes = pickle.load(infile)
    #     infile.close()

    print(len(correct_custom_clashes), len(incorrect_custom_clashes))

    fig, ax = plt.subplots()
    sns.distplot(incorrect_custom_clashes, hist=True, label="intolerable clash")
    sns.distplot(correct_custom_clashes, hist=True, label="tolerable clash")
    plt.title('Clash Distributions for custom clash function')
    plt.xlabel('clash volume')
    plt.ylabel('frequency')
    ax.legend()
    fig.savefig('custom_clash.png')

    fig, ax = plt.subplots()
    sns.distplot(incorrect_schrod_clashes, hist=True, label="intolerable clash")
    sns.distplot(correct_schrod_clashes, hist=True, label="tolerable clash")
    plt.title('Clash Distributions for custom schrod function')
    plt.xlabel('clash volume')
    plt.ylabel('frequency')
    ax.legend()
    fig.savefig('schrod_clash.png')


if __name__ == "__main__":
    main()
