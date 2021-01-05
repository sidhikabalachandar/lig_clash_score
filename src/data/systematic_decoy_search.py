"""
The purpose of this code is to first create the raw directory folder and include the following files
starting protein receptor
starting ligand
target ligand
glide pose viewer file

Then the top glide poses are added

Then the decoys are created

It can be run on sherlock using
$ $SCHRODINGER/run python3 systematic_decoy_search.py get_rmsd /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw --protein P22894 --target 3tt4 --start 1zvx --rotation_search_step_size 5 --index 0
"""

import argparse
import os
import schrodinger.structure as structure
import subprocess
import schrodinger.structutils.transform as transform
from schrodinger.structutils.transform import get_centroid
import schrodinger.structutils.interactions.steric_clash as steric_clash
import schrodinger.structutils.rmsd as rmsd
import schrodinger.structutils.build as build
import math
import time
import statistics
import random
from tqdm import tqdm
import pandas as pd
import scipy.spatial
import numpy as np

_CONFGEN_CMD = ("$SCHRODINGER/confgenx -WAIT -optimize -drop_problematic -num_conformers {num_conformers} "
                "-max_num_conformers {num_conformers} {input_file}")
_ALIGN_CMD = "$SCHRODINGER/shape_screen -shape {shape} -screen {screen} -WAIT -JOB {job_name}"
X_AXIS = [1.0, 0.0, 0.0]  # x-axis unit vector
Y_AXIS = [0.0, 1.0, 0.0]  # y-axis unit vector
Z_AXIS = [0.0, 0.0, 1.0]  # z-axis unit vector


def run_cmd(cmd, error_msg=None, raise_except=False):
    try:
        return subprocess.check_output(
            cmd,
            universal_newlines=True,
            shell=True)
    except Exception as e:
        if error_msg is not None:
            print(error_msg)
        if raise_except:
            raise e


def get_conformer_groups(n, target, start, protein, raw_root):
    pair = '{}-to-{}'.format(target, start)
    protein_path = os.path.join(raw_root, protein)
    pair_path = os.path.join(protein_path, pair)
    conformer_file = os.path.join(pair_path, "{}_lig0-out.maegz".format(target))
    conformers = list(structure.StructureReader(conformer_file))
    grouped_files = []

    for i in range(0, len(conformers), n):
        grouped_files.append(conformers[i: i + n])

    return grouped_files


def get_grid_groups(grid_size, n):
    grid = []
    for dx in range(-grid_size, grid_size):
        for dy in range(-grid_size, grid_size):
            for dz in range(-grid_size, grid_size):
                grid.append([dx, dy, dz])

    grouped_files = []

    for i in range(0, len(grid), n):
        grouped_files += [grid[i: i + n]]

    return grouped_files


def get_prots(docked_prot_file):
    """
    gets list of all protein, target ligands, and starting ligands in the index file
    :param docked_prot_file: (string) file listing proteins to process
    :return: process (list) list of all protein, target ligands, and starting ligands to process
    """
    process = []
    with open(docked_prot_file) as fp:
        for line in tqdm(fp, desc='index file'):
            if line[0] == '#': continue
            protein, target, start = line.strip().split()
            process.append((protein, target, start))

    return process


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

def transform_structure(st, matrix):
    """
    Transforms atom coordinates of the structure using a 4x4
    transformation matrix.

    st (structure.Structure)

    matrix (numpy.array)
        4x4 numpy array representation of transformation matrix.

    """

    # Modifying this array will directly alter the actual coordinates:
    atom_xyz_array = np.array(st.getXYZ(copy=False))
    num_atoms = len(atom_xyz_array)
    ones = np.ones((num_atoms, 1))
    atom_xyz_array = np.concatenate((atom_xyz_array, ones), axis=1)
    atom_xyz_array = np.resize(atom_xyz_array, (num_atoms, 4, 1))
    atom_xyz_array = np.matmul(matrix, atom_xyz_array)
    atom_xyz_array = np.resize(atom_xyz_array, (num_atoms, 4))
    atom_xyz_array = atom_xyz_array[:, 0:3]
    st.setXYZ(atom_xyz_array[:, 0:3])


def translate_structure(st, x, y, z):
    trans_matrix = get_translation_matrix([x, y, z])
    transform_structure(st, trans_matrix)


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


def rotate_structure(st, x_angle, y_angle, z_angle, rot_center):
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
    transform_structure(st, combined_rot_matrix)


def time_conformer_decoys(pair_path, start_lig_center, target_lig, prot, rotation_search_step_size):
    translate_times = []
    rotate_times = []
    conformer_file = os.path.join(pair_path, "aligned_to_start_without_hydrogen_conformers.mae")
    conformers = list(structure.StructureReader(conformer_file))

    for conformer in conformers:
        conformer_center = list(get_centroid(conformer))

        # translation
        grid_loc = [0, 0, 0]
        start = time.time()
        transform.translate_structure(conformer, start_lig_center[0] - conformer_center[0] + grid_loc[0],
                                      start_lig_center[1] - conformer_center[1] + grid_loc[1],
                                      start_lig_center[2] - conformer_center[2] + grid_loc[2])
        end = time.time()
        translate_times.append(end - start)

        # rotation
        start = time.time()
        transform.rotate_structure(conformer, math.radians(-30 - rotation_search_step_size), 0, 0, conformer_center)
        end = time.time()
        rotate_times.append(end - start)

    print("Average schrodinger translate time =", statistics.mean(translate_times))
    print("Average schrodinger rotate time =", statistics.mean(rotate_times))

    translate_times = []
    rotate_times = []
    conformer_file = os.path.join(pair_path, "aligned_to_start_without_hydrogen_conformers.mae")
    conformers = list(structure.StructureReader(conformer_file))

    for conformer in conformers:
        conformer_center = list(get_centroid(conformer))

        # translation
        grid_loc = [0, 0, 0]
        start = time.time()
        translate_structure(conformer, start_lig_center[0] - conformer_center[0] + grid_loc[0],
                            start_lig_center[1] - conformer_center[1] + grid_loc[1],
                            start_lig_center[2] - conformer_center[2] + grid_loc[2])
        end = time.time()
        translate_times.append(end - start)

        # rotation
        start = time.time()
        rotate_structure(conformer, math.radians(-30 - rotation_search_step_size), 0, 0, conformer_center)
        end = time.time()
        rotate_times.append(end - start)

    print("Average custom translate time =", statistics.mean(translate_times))
    print("Average custom rotate time =", statistics.mean(rotate_times))

    clash_iterator_times = []
    clash_volume_times = []
    rmsd_times = []
    rotation_search_step_size_rad = math.radians(rotation_search_step_size)

    conformer_file = os.path.join(pair_path, "aligned_to_start_without_hydrogen_conformers.mae")
    conformers = list(structure.StructureReader(conformer_file))
    for conformer in conformers:
        conformer_center = list(get_centroid(conformer))

        # translation
        grid_loc = [0, 0, 0]
        translate_structure(conformer, start_lig_center[0] - conformer_center[0] + grid_loc[0],
                            start_lig_center[1] - conformer_center[1] + grid_loc[1],
                            start_lig_center[2] - conformer_center[2] + grid_loc[2])
        conformer_center = list(get_centroid(conformer))

        # keep track of rotation angles
        rotate_structure(conformer, math.radians(-30 - rotation_search_step_size), 0, 0, conformer_center)
        x_so_far = -30 - rotation_search_step_size
        y_so_far = 0
        z_so_far = 0

        for _ in range(-30, 30, rotation_search_step_size):
            # x rotation
            rotate_structure(conformer, rotation_search_step_size_rad,
                                       math.radians(-30 - rotation_search_step_size - y_so_far), 0, conformer_center)
            x_so_far += 1
            y_so_far += -30 - rotation_search_step_size - y_so_far

            for _ in range(-30, 30, rotation_search_step_size):
                # y rotation
                rotate_structure(conformer, 0, rotation_search_step_size_rad,
                                           math.radians(-30 - rotation_search_step_size - z_so_far), conformer_center)
                y_so_far += 1
                z_so_far += -30 - rotation_search_step_size - z_so_far

                for _ in range(-30, 30, rotation_search_step_size):
                    # z rotation
                    rotate_structure(conformer, 0, 0, rotation_search_step_size_rad, conformer_center)
                    z_so_far += 1

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

                    if len(clash_iterator_times) == 1000:
                        print("Average clash iterator time =", statistics.mean(clash_iterator_times))
                        print("Average clash volume time =", statistics.mean(clash_volume_times))
                        print("Average rmsd time =", statistics.mean(rmsd_times))
                        return


def create_conformer_decoys(grid, start_lig_center, target_lig, cutoff, rotation_search_step_size, protein,
                            target, start, index, pair_path, test, x_rot, y_rot, z_rot):
    counter = 0
    data_dict = {'protein': [], 'target': [], 'start': [], 'num_conformers': [], 'num_poses_searched': [],
                 'num_correct_poses_found': [], 'time_elapsed': [], 'time_elapsed_per_conformer': [], 'grid_loc_x': [],
                 'grid_loc_y': [], 'grid_loc_z': []}

    for grid_loc in grid:
        num_correct_found = 0
        conformer_file = os.path.join(pair_path, "aligned_to_start_without_hydrogen_conformers.mae")
        conformers = list(structure.StructureReader(conformer_file))
        decoy_start_time = time.time()
        conformer = conformers[248]
        rmsd_val = rmsd.calculate_in_place_rmsd(conformer, conformer.getAtomIndices(),
                                                target_lig, target_lig.getAtomIndices())
        print("aligned conformer rmsd:", rmsd_val)
        # for conformer in conformers:
        conformer_center = list(get_centroid(conformer))

        # translation
        translate_structure(conformer, grid_loc[0], grid_loc[1], grid_loc[2])
        rmsd_val = rmsd.calculate_in_place_rmsd(conformer, conformer.getAtomIndices(),
                                                target_lig, target_lig.getAtomIndices())
        print("translated conformer rmsd:", rmsd_val)
        conformer_center = list(get_centroid(conformer))

        print("Before:", 0, 0, 0)
        print("x + ", -rotation_search_step_size)
        # keep track of rotation angles
        rotate_structure(conformer, math.radians(-rotation_search_step_size), 0, 0, conformer_center)
        x_so_far = -rotation_search_step_size
        y_so_far = 0
        z_so_far = 0
        print("After:", x_so_far, y_so_far, z_so_far)

        for _ in range(0, 30, rotation_search_step_size):
            print("Before:", x_so_far, y_so_far, z_so_far)
            print("x + ", rotation_search_step_size, "y + ", -rotation_search_step_size - y_so_far)
            # x rotation
            rotate_structure(conformer, math.radians(rotation_search_step_size),
                                       math.radians(-rotation_search_step_size - y_so_far), 0, conformer_center)
            x_so_far += rotation_search_step_size
            y_so_far += -rotation_search_step_size - y_so_far
            print("After:", x_so_far, y_so_far, z_so_far)

            for _ in range(0, 30, rotation_search_step_size):
                print("Before:", x_so_far, y_so_far, z_so_far)
                print("y + ", rotation_search_step_size, "z + ", -rotation_search_step_size - z_so_far)
                # y rotation
                rotate_structure(conformer, 0, math.radians(rotation_search_step_size),
                                           math.radians(-rotation_search_step_size - z_so_far), conformer_center)
                y_so_far += rotation_search_step_size
                z_so_far += -rotation_search_step_size - z_so_far
                print("After:", x_so_far, y_so_far, z_so_far)

                for _ in range(0, 30, rotation_search_step_size):
                    print("Before:", x_so_far, y_so_far, z_so_far)
                    print("z + ", rotation_search_step_size)
                    # z rotation
                    rotate_structure(conformer, 0, 0, math.radians(rotation_search_step_size), conformer_center)
                    z_so_far += rotation_search_step_size
                    counter += 1
                    print("After:", x_so_far, y_so_far, z_so_far)
                    print("CALCULATE RMSD")

                    if test and x_rot == x_so_far and y_rot == y_so_far and z_rot == z_so_far:
                        return conformer

                    rmsd_val = rmsd.calculate_in_place_rmsd(conformer, conformer.getAtomIndices(),
                                                            target_lig, target_lig.getAtomIndices())
                    if x_so_far == 0 and y_so_far == 0 and z_so_far == 0:
                        print("No rotation pose rmsd:", rmsd_val)
                    if rmsd_val < cutoff:
                        num_correct_found += 1

        # keep track of rotation angles
        x_so_far = 0
        y_so_far = 0
        z_so_far = 0

        for _ in range(-rotation_search_step_size, -30, -rotation_search_step_size):
            print("Before:", x_so_far, y_so_far, z_so_far)
            print("x + ", -rotation_search_step_size, "y + ", -y_so_far)
            # x rotation
            rotate_structure(conformer, math.radians(-rotation_search_step_size),
                             math.radians(-y_so_far), 0, conformer_center)
            x_so_far += -rotation_search_step_size
            y_so_far += -y_so_far
            print("After:", x_so_far, y_so_far, z_so_far)

            for _ in range(-rotation_search_step_size, -30, -rotation_search_step_size):
                print("Before:", x_so_far, y_so_far, z_so_far)
                print("y + ", -rotation_search_step_size, "z + ", -z_so_far)
                # y rotation
                rotate_structure(conformer, 0, math.radians(-rotation_search_step_size),
                                 math.radians(-z_so_far), conformer_center)
                y_so_far += -rotation_search_step_size
                z_so_far += -z_so_far
                print("After:", x_so_far, y_so_far, z_so_far)

                for _ in range(-rotation_search_step_size, -30, -rotation_search_step_size):
                    print("Before:", x_so_far, y_so_far, z_so_far)
                    print("z + ", -rotation_search_step_size)
                    # z rotation
                    rotate_structure(conformer, 0, 0, math.radians(-rotation_search_step_size), conformer_center)
                    z_so_far += -rotation_search_step_size
                    counter += 1
                    print("After:", x_so_far, y_so_far, z_so_far)
                    print("CALCULATE RMSD")

                    if test and x_rot == x_so_far and y_rot == y_so_far and z_rot == z_so_far:
                        return conformer

                    rmsd_val = rmsd.calculate_in_place_rmsd(conformer, conformer.getAtomIndices(),
                                                            target_lig, target_lig.getAtomIndices())
                    if x_so_far == 0 and y_so_far == 0 and z_so_far == 0:
                        print("No rotation pose rmsd:", rmsd_val)
                        return
                    if rmsd_val < cutoff:
                        num_correct_found += 1

        # print("Before:", 0, 0, 0)
        # print("x + ", - 30 - rotation_search_step_size)
        # # keep track of rotation angles
        # rotate_structure(conformer, math.radians(-30 - rotation_search_step_size), 0, 0, conformer_center)
        # x_so_far = -30 - rotation_search_step_size
        # y_so_far = 0
        # z_so_far = 0
        # print("After:", x_so_far, y_so_far, z_so_far)
        #
        # for _ in range(-30, 30, rotation_search_step_size):
        #     print("Before:", x_so_far, y_so_far, z_so_far)
        #     print("x + ", rotation_search_step_size, "y + ", -30 - rotation_search_step_size - y_so_far)
        #     # x rotation
        #     rotate_structure(conformer, math.radians(rotation_search_step_size),
        #                                math.radians(-30 - rotation_search_step_size - y_so_far), 0, conformer_center)
        #     x_so_far += rotation_search_step_size
        #     y_so_far += -30 - rotation_search_step_size - y_so_far
        #     print("After:", x_so_far, y_so_far, z_so_far)
        #
        #     for _ in range(-30, 30, rotation_search_step_size):
        #         print("Before:", x_so_far, y_so_far, z_so_far)
        #         print("y + ", rotation_search_step_size, "z + ", -30 - rotation_search_step_size - z_so_far)
        #         # y rotation
        #         rotate_structure(conformer, 0, math.radians(rotation_search_step_size),
        #                                    math.radians(-30 - rotation_search_step_size - z_so_far), conformer_center)
        #         y_so_far += rotation_search_step_size
        #         z_so_far += -30 - rotation_search_step_size - z_so_far
        #         print("After:", x_so_far, y_so_far, z_so_far)
        #
        #         for _ in range(-30, 30, rotation_search_step_size):
        #             print("Before:", x_so_far, y_so_far, z_so_far)
        #             print("z + ", rotation_search_step_size)
        #             # z rotation
        #             rotate_structure(conformer, 0, 0, math.radians(rotation_search_step_size), conformer_center)
        #             z_so_far += rotation_search_step_size
        #             counter += 1
        #             print("After:", x_so_far, y_so_far, z_so_far)
        #             print("CALCULATE RMSD")
        #
        #             if test and x_rot == x_so_far and y_rot == y_so_far and z_rot == z_so_far:
        #                 return conformer
        #
        #             rmsd_val = rmsd.calculate_in_place_rmsd(conformer, conformer.getAtomIndices(),
        #                                                     target_lig, target_lig.getAtomIndices())
        #             if x_so_far == 0 and y_so_far == 0 and z_so_far == 0:
        #                 print("No rotation pose rmsd:", rmsd_val)
        #                 return
        #             if rmsd_val < cutoff:
        #                 num_correct_found += 1

        decoy_end_time = time.time()

        data_dict['protein'].append(protein)
        data_dict['target'].append(target)
        data_dict['start'].append(start)
        data_dict['num_conformers'].append(len(conformers))
        data_dict['num_poses_searched'].append(counter)
        data_dict['num_correct_poses_found'].append(num_correct_found)
        data_dict['time_elapsed'].append(decoy_end_time - decoy_start_time)
        data_dict['time_elapsed_per_conformer'].append((decoy_end_time - decoy_start_time) / len(conformers))
        data_dict['grid_loc_x'].append(grid_loc[0])
        data_dict['grid_loc_y'].append(grid_loc[1])
        data_dict['grid_loc_z'].append(grid_loc[2])

    # df = pd.DataFrame.from_dict(data_dict)
    # save_folder = os.path.join(os.getcwd(), 'decoy_timing_data', '{}_{}-to-{}'.format(protein, target, start))
    # if not os.path.exists(save_folder):
    #     os.mkdir(save_folder)
    # df.to_csv(os.path.join(save_folder, '{}.csv'.format(index)))
    # return None


def run_conformer_all(process, raw_root, run_path, docked_prot_file):
    counter = 0
    for protein, target, start in process:
        if counter == 10:
            break
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, pair)
        conformer_file = os.path.join(pair_path, "{}_lig0-out.maegz".format(target))
        if not os.path.exists(conformer_file):
            cmd = 'sbatch -p owners -t 1:00:00 -o {} --wrap="$SCHRODINGER/run python3 systematic_decoy_search.py ' \
                  'conformer_group {} {} {} --protein {} --target {} --start {}"'
            os.system(cmd.format(os.path.join(run_path, 'conformer_{}_{}-to-{}.out'.format(protein, target, start)),
                                 docked_prot_file, run_path, raw_root, protein, target, start))
            counter += 1
        else:
            conformers = list(structure.StructureReader(conformer_file))
            if len(conformers) > 1:
                counter += 1
            print("found:", protein, target, start)


def gen_ligand_conformers(path, output_dir, num_conformers):
    current_dir = os.getcwd()
    os.chdir(output_dir)
    basename = os.path.basename(path)
    ### Note: For some reason, confgen isn't able to find the .mae file,
    # unless it is in working directory. So, we need to copy it over.
    ### Note: There may be duplicated ligand names (for different targets).
    # Since it only happens for CHEMBL ligand, just ignore it for now.
    # Otherwise, might want consider to generate the conformers to separate
    # folders for each (target, ligand) pair.
    # Run ConfGen
    run_cmd(f'cp {path:} ./{basename:}')
    command = _CONFGEN_CMD.format(num_conformers=num_conformers,
                                  input_file=f'./{basename:}')
    run_cmd(command, f'Failed to run ConfGen on {path:}')
    run_cmd(f'rm ./{basename:}')
    os.chdir(current_dir)


def run_conformer_check(process, raw_root):
    unfinished = []
    counter = 0
    for protein, target, start in process:
        if counter == 10:
            break
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, pair)
        if not os.path.exists(os.path.join(pair_path, '{}_lig0-out.maegz'.format(target))):
            process.append((protein, target, start))
        else:
            conformer_file = os.path.join(pair_path, "{}_lig0-out.maegz".format(target))
            conformers = list(structure.StructureReader(conformer_file))
            print(protein, target, start, len(conformers), counter)
            if len(conformers) > 1:
                counter += 1

    print("Missing:", len(unfinished), "/ 10")
    print(unfinished)


def run_align_all(process, raw_root, run_path, docked_prot_file, n):
    counter = 0
    for protein, target, start in process:
        if counter == 10:
            break
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, pair)

        conformer_file = os.path.join(pair_path, "{}_lig0-out.maegz".format(target))
        conformers = list(structure.StructureReader(conformer_file))
        if len(conformers) == 1:
            continue
        else:
            counter += 1

        output_path = os.path.join(pair_path, 'conformers')
        if not os.path.exists(os.path.join(output_path, '0_align_without_hydrogen.mae')):
            print(protein, target, start, counter)
            grouped_files = get_conformer_groups(n, target, start, protein, raw_root)
            for i, group in enumerate(grouped_files):
                cmd = 'sbatch -p owners -t 1:00:00 -o {} --wrap="$SCHRODINGER/run python3 systematic_decoy_search.py ' \
                      'align_group {} {} {} --n {} --index {} --protein {} --target {} --start {}"'
                os.system(cmd.format(os.path.join(run_path, 'decoy{}.out'.format(i)), docked_prot_file, run_path,
                                     raw_root, n, i, protein, target, start))


def run_align_group(grouped_files, index, n, protein, target, start, raw_root):
    for i, conformer in enumerate(grouped_files[index]):
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, pair)
        output_path = os.path.join(pair_path, 'conformers')
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        num = str(n * index + i)
        os.chdir(output_path)
        screen_file = os.path.join(output_path, "screen_{}.mae".format(num))
        with structure.StructureWriter(screen_file) as screen:
            screen.append(conformer)
        shape_file = os.path.join(pair_path, '{}_lig.mae'.format(start))
        run_cmd(_ALIGN_CMD.format(shape=shape_file, screen=screen_file, job_name=num))
        aligned_conformer_file = os.path.join(output_path, '{}_align.maegz'.format(num))
        aligned_conformer = list(structure.StructureReader(aligned_conformer_file))[0]
        build.delete_hydrogens(aligned_conformer)
        no_hydrogen_file = os.path.join(output_path, "{}_align_without_hydrogen.mae".format(num))
        with structure.StructureWriter(no_hydrogen_file) as no_h:
            no_h.append(aligned_conformer)


def run_align_check(process, raw_root):
    unfinished = []
    counter = 0
    for protein, target, start in process:
        if counter == 10:
            break
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, pair)

        conformer_file = os.path.join(pair_path, "{}_lig0-out.maegz".format(target))
        conformers = list(structure.StructureReader(conformer_file))
        print(protein, target, start, counter)
        if len(conformers) == 1:
            continue
        else:
            counter += 1
        output_path = os.path.join(pair_path, 'conformers')

        for i in range(len(conformers)):
            if not os.path.exists(os.path.join(output_path, '{}_align_without_hydrogen.mae'.format(i))):
                unfinished.append((protein, target, start, i))

    print("Missing:", len(unfinished))
    print(unfinished)


def run_align_combine(process, raw_root):
    counter = 0
    for protein, target, start in process:
        if counter == 10:
            break
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, pair)
        conformer_file = os.path.join(pair_path, "{}_lig0-out.maegz".format(target))
        conformers = list(structure.StructureReader(conformer_file))
        if len(conformers) == 1:
            continue
        else:
            counter += 1
        output_path = os.path.join(pair_path, 'conformers')

        combined_file = os.path.join(pair_path, "aligned_to_start_without_hydrogen_conformers.mae")
        with structure.StructureWriter(combined_file) as combined:
            for i in range(len(conformers)):
                aligned_file = os.path.join(output_path, '{}_align_without_hydrogen.mae'.format(i))
                s = list(structure.StructureReader(aligned_file))[0]
                combined.append(s)

        print(len(list(structure.StructureReader(combined_file))))


def search_system_caller(process, raw_root, run_path, docked_prot_file, rotation_search_step_size, grouped_files):
    counter = 0
    for protein, target, start in process:
        if counter == 10:
            break
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, pair)
        conformer_file = os.path.join(pair_path, "{}_lig0-out.maegz".format(target))
        conformers = list(structure.StructureReader(conformer_file))
        if len(conformers) == 1:
            continue
        else:
            counter += 1
        for i in range(len(grouped_files)):
            cmd = 'sbatch -p owners -t 3:00:00 -o {} --wrap="$SCHRODINGER/run python3 systematic_decoy_search.py ' \
                  'search {} {} {} --protein {} --target {} --start {} --rotation_search_step_size {} --index {}"'
            out_file_name = '{}_{}-to-{}_{}.out'.format(protein, target, start, i)
            os.system(cmd.format(os.path.join(run_path, out_file_name), docked_prot_file, run_path, raw_root, protein,
                                 target, start, rotation_search_step_size, i))


def run_search(protein, target, start, index, raw_root, get_time, cutoff, rotation_search_step_size, grid, no_prot_h,
               pocket_only, test=False, x_rot=0, y_rot=0, z_rot=0):
    pair = '{}-to-{}'.format(target, start)
    protein_path = os.path.join(raw_root, protein)
    pair_path = os.path.join(protein_path, pair)
    start_lig_file = os.path.join(pair_path, '{}_lig.mae'.format(start))
    start_lig = list(structure.StructureReader(start_lig_file))[0]
    target_lig_file = os.path.join(pair_path, 'ligand_poses', '{}_lig0.mae'.format(target))
    target_lig = list(structure.StructureReader(target_lig_file))[0]
    build.delete_hydrogens(target_lig)
    start_lig_center = list(get_centroid(start_lig))
    prot_file = os.path.join(pair_path, '{}_prot.mae'.format(start))
    prot = list(structure.StructureReader(prot_file))[0]
    print(prot.atom_total)
    if pocket_only:
        get_pocket_res(prot, target_lig, 6)
        print(prot.atom_total)
    if no_prot_h:
        build.delete_hydrogens(prot)
        print(prot.atom_total)

    if get_time:
        time_conformer_decoys(pair_path, start_lig_center, target_lig, prot, rotation_search_step_size)
    else:
        conformer = create_conformer_decoys(grid, start_lig_center, target_lig, cutoff, rotation_search_step_size,
                                            protein, target, start, index, pair_path, test, x_rot, y_rot, z_rot)
        return conformer


def run_test_search(protein, target, start, raw_root, cutoff, rotation_search_step_size, pair_path, no_prot_h,
                    pocket_only, get_time):
    angles = [i for i in range(-30, 30, rotation_search_step_size)]
    angles = angles[:5]
    x_rot = random.choice(angles)
    y_rot = random.choice(angles)
    z_rot = random.choice(angles)

    conformer = run_search(protein, target, start, raw_root, get_time, cutoff, rotation_search_step_size, no_prot_h,
                           pocket_only, True, x_rot, y_rot, z_rot)
    start_lig_file = os.path.join(pair_path, '{}_lig.mae'.format(start))
    start_lig = list(structure.StructureReader(start_lig_file))[0]
    start_lig_center = list(get_centroid(start_lig))

    conformer_file = os.path.join(pair_path, "aligned_to_start_conformers.mae")
    base_conf = list(structure.StructureReader(conformer_file))[0]
    grid_loc = [0, 0, 0]
    base_conf_center = list(get_centroid(base_conf))
    translate_structure(base_conf, start_lig_center[0] - base_conf_center[0] + grid_loc[0],
                                  start_lig_center[1] - base_conf_center[1] + grid_loc[1],
                                  start_lig_center[2] - base_conf_center[2] + grid_loc[2])
    base_conf_center = list(get_centroid(base_conf))
    rotate_structure(base_conf, math.radians(x_rot), math.radians(y_rot), math.radians(z_rot),
                               base_conf_center)
    rmsd_val = rmsd.calculate_in_place_rmsd(conformer, conformer.getAtomIndices(), base_conf,
                                            base_conf.getAtomIndices())
    if abs(rmsd_val) < 0.2:
        print("Search works properly", rmsd_val)
    else:
        print("x_rot =", x_rot, "y_rot =", y_rot, "z_rot =", z_rot)
        print("RMSD =", rmsd_val, "but RMSD should equal 0")


def run_combine_search_data(process, grouped_files, raw_root, grid_size):
    counter = 0
    dfs = []
    protein_pairs = []
    for protein, target, start in process:
        if counter == 10:
            break
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, pair)
        conformer_file = os.path.join(pair_path, "{}_lig0-out.maegz".format(target))
        conformers = list(structure.StructureReader(conformer_file))
        if len(conformers) == 1:
            continue
        else:
            counter += 1
        protein_pairs.append((protein, target, start))
        save_folder = os.path.join(os.getcwd(), 'decoy_timing_data', '{}_{}-to-{}'.format(protein, target, start))
        for i in range(len(grouped_files)):
            dfs.append(pd.read_csv(os.path.join(save_folder, '{}.csv'.format(i))))

    data_df = pd.concat(dfs)
    pair_dict = {}

    for protein, target, start in protein_pairs:
        pair = '{}_{}-to-{}'.format(protein, target, start)
        pair_df = data_df[(data_df['protein'] == protein) & (data_df['target'] == target) & (data_df['start'] == start)]
        pair_dict[pair] = []
        for i in range(grid_size + 1):
            grid_df = pair_df[(pair_df['grid_loc_x'] <= i) & (pair_df['grid_loc_x'] >= -i) &
                              (pair_df['grid_loc_y'] <= i) & (pair_df['grid_loc_y'] >= -i) &
                              (pair_df['grid_loc_z'] <= i) & (pair_df['grid_loc_z'] >= -i)]
            pair_dict[pair].append(grid_df.sum(axis=0)['num_correct_poses_found'])

    print(pair_dict)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either align or search')
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('--protein', type=str, default='', help='protein name')
    parser.add_argument('--target', type=str, default='', help='target ligand name')
    parser.add_argument('--start', type=str, default='', help='start ligand name')
    parser.add_argument('--align_n', type=int, default=10, help='number of alignments processed in each job')
    parser.add_argument('--rotation_search_step_size', type=int, default=1, help='step size between each angle '
                                                                                 'checked, in degrees')
    parser.add_argument('--index', type=int, default=-1, help='grid point group index')
    parser.add_argument('--rmsd_cutoff', type=int, default=2, help='rmsd accuracy cutoff between predicted ligand pose '
                                                                   'and true ligand pose')
    parser.add_argument('--num_conformers', type=int, default=300, help='maximum number of conformers considered')
    parser.add_argument('--grid_size', type=int, default=6, help='grid size in positive and negative x, y, z '
                                                                 'directions')
    parser.add_argument('--grid_n', type=int, default=30, help='number of grid_points processed in each job')
    parser.add_argument('--time', dest='get_time', action='store_true')
    parser.add_argument('--no_time', dest='get_time', action='store_false')
    parser.set_defaults(get_time=False)
    parser.add_argument('--remove_prot_h', dest='no_prot_h', action='store_true')
    parser.add_argument('--keep_prot_h', dest='no_prot_h', action='store_false')
    parser.set_defaults(no_prot_h=False)
    parser.add_argument('--prot_pocket_only', dest='pocket_only', action='store_true')
    parser.add_argument('--all_prot', dest='pocket_only', action='store_false')
    parser.set_defaults(pocket_only=False)

    args = parser.parse_args()

    random.seed(0)

    if not os.path.exists(args.run_path):
        os.mkdir(args.run_path)

    pair = '{}-to-{}'.format(args.target, args.start)
    protein_path = os.path.join(args.raw_root, args.protein)
    pair_path = os.path.join(protein_path, pair)

    if args.task == 'conformer_all':
        process = get_prots(args.docked_prot_file)
        random.shuffle(process)
        run_conformer_all(process, args.raw_root, args.run_path, args.docked_prot_file)

    elif args.task == 'conformer_group':
        target_lig_file = os.path.join(pair_path, 'ligand_poses', '{}_lig0.mae'.format(args.target))
        gen_ligand_conformers(target_lig_file, pair_path, args.num_conformers)
        if os.path.exists(os.path.join(pair_path, '{}_lig0.log'.format(args.target))):
            os.remove(os.path.join(pair_path, '{}_lig0.log'.format(args.target)))

    if args.task == 'conformer_check':
        process = get_prots(args.docked_prot_file)
        random.shuffle(process)
        run_conformer_check(process, args.raw_root)

    if args.task == 'align_all':
        process = get_prots(args.docked_prot_file)
        random.shuffle(process)
        run_align_all(process, args.raw_root, args.run_path, args.docked_prot_file, args.align_n)

    elif args.task == 'align_group':
        grouped_files = get_conformer_groups(args.align_n, args.target, args.start, args.protein, args.raw_root)
        run_align_group(grouped_files, args.index, args.n, args.protein, args.target, args.start, args.raw_root)

    elif args.task == 'align_check':
        process = get_prots(args.docked_prot_file)
        random.shuffle(process)
        run_align_check(process, args.raw_root)

    elif args.task == 'align_combine':
        process = get_prots(args.docked_prot_file)
        random.shuffle(process)
        run_align_combine(process, args.raw_root)

    elif args.task == 'run_search':
        process = get_prots(args.docked_prot_file)
        random.shuffle(process)
        grouped_files = get_grid_groups(args.grid_size, args.grid_n)
        search_system_caller(process, args.raw_root, args.run_path, args.docked_prot_file,
                             args.rotation_search_step_size, grouped_files)

    elif args.task == 'search':
        # grouped_files = get_grid_groups(args.grid_size, args.grid_n)
        # run_search(args.protein, args.target, args.start, args.index, args.raw_root, args.get_time, args.rmsd_cutoff,
        #            args.rotation_search_step_size, grouped_files[args.index], args.no_prot_h, args.pocket_only)
        run_search(args.protein, args.target, args.start, args.index, args.raw_root, args.get_time, args.rmsd_cutoff,
                   args.rotation_search_step_size, [[0, 0, 0]], args.no_prot_h, args.pocket_only)

    elif args.task == 'check_search':
        process = get_prots(args.docked_prot_file)
        random.shuffle(process)
        grouped_files = get_grid_groups(args.grid_size, args.grid_n)
        counter = 0
        unfinished = []
        for protein, target, start in process:
            if counter == 10:
                break
            pair = '{}-to-{}'.format(target, start)
            protein_path = os.path.join(args.raw_root, protein)
            pair_path = os.path.join(protein_path, pair)
            conformer_file = os.path.join(pair_path, "{}_lig0-out.maegz".format(target))
            conformers = list(structure.StructureReader(conformer_file))
            if len(conformers) == 1:
                continue
            else:
                counter += 1
            save_folder = os.path.join(os.getcwd(), 'decoy_timing_data', '{}_{}-to-{}'.format(protein, target, start))
            for i in range(len(grouped_files)):
                if not os.path.exists(os.path.join(save_folder, '{}.csv'.format(i))):
                    unfinished.append((protein, target, start, i))
        print("Missing:", len(unfinished))
        print(unfinished)

    elif args.task == 'test_search':
        run_test_search(args.protein, args.target, args.start, args.raw_root, args.rmsd_cutoff,
                        args.rotation_search_step_size, pair_path, args.no_prot_h, args.pocket_only, args.get_time)

    elif args.task == 'combine_search_data':
        process = get_prots(args.docked_prot_file)
        random.shuffle(process)
        grouped_files = get_grid_groups(args.grid_size, args.grid_n)
        run_combine_search_data(process, grouped_files, args.raw_root, args.grid_size)

    elif args.task == 'get_dist':
        process = get_prots(args.docked_prot_file)
        random.shuffle(process)
        counter = 0
        for protein, target, start in process:
            if counter == 10:
                break
            pair = '{}-to-{}'.format(target, start)
            protein_path = os.path.join(args.raw_root, protein)
            pair_path = os.path.join(protein_path, pair)
            conformer_file = os.path.join(pair_path, "{}_lig0-out.maegz".format(target))
            conformers = list(structure.StructureReader(conformer_file))
            if len(conformers) == 1:
                continue
            else:
                counter += 1
            start_lig_file = os.path.join(pair_path, '{}_lig.mae'.format(start))
            start_lig = list(structure.StructureReader(start_lig_file))[0]
            start_lig_center = list(get_centroid(start_lig))
            target_lig_file = os.path.join(pair_path, 'ligand_poses', '{}_lig0.mae'.format(target))
            target_lig = list(structure.StructureReader(target_lig_file))[0]
            target_lig_center = list(get_centroid(target_lig))
            dist = math.sqrt(((start_lig_center[0] - target_lig_center[0]) ** 2) +
                             ((start_lig_center[1] - target_lig_center[1]) ** 2) +
                             ((start_lig_center[2] - target_lig_center[2]) ** 2))
            print(protein, target, start, dist)


    elif args.task == 'test_rotate_translate':
        prot_file = os.path.join(pair_path, '{}_prot.mae'.format(args.start))
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
        rotate_structure(custom_prot, rotation_vector[0], rotation_vector[1], rotation_vector[2], rotation_center)
        schrodinger_atoms = np.array(schrodinger_prot.getXYZ(copy=False))
        custom_atoms = np.array(custom_prot.getXYZ(copy=False))
        if np.amax(np.absolute(schrodinger_atoms - custom_atoms)) < 10 ** -7:
            print("Rotate function works properly")
        else:
            print("Error in rotate function")

    elif args.task == 'get_rmsd':
        conformer_file = os.path.join(pair_path, "aligned_to_start_without_hydrogen_conformers.mae".format(args.target))
        conformers = list(structure.StructureReader(conformer_file))

        target_lig_file = os.path.join(pair_path, 'ligand_poses', '{}_lig0.mae'.format(args.target))
        target_lig = list(structure.StructureReader(target_lig_file))[0]
        build.delete_hydrogens(target_lig)
        start_lig_file = os.path.join(pair_path, '{}_lig.mae'.format(args.start))
        start_lig = list(structure.StructureReader(start_lig_file))[0]
        start_lig_center = list(get_centroid(start_lig))


        rmsds = []
        for i, conformer in tqdm(enumerate(conformers), desc='going through conformers'):
            conformer_center = list(get_centroid(conformer))
            translate_structure(conformer, start_lig_center[0] - conformer_center[0],
                                start_lig_center[1] - conformer_center[1],
                                start_lig_center[2] - conformer_center[2])
            rmsds.append((conformer, rmsd.calculate_in_place_rmsd(conformer, conformer.getAtomIndices(), target_lig,
                                                          target_lig.getAtomIndices()), i))

        # best_match_conformer = min(rmsds, key=lambda x: x[1])
        # print(best_match_conformer[1], best_match_conformer[2])
        # file = os.path.join(pair_path, 'best_match_conformer.mae')
        # with structure.StructureWriter(file) as best_match:
        #     best_match.append(best_match_conformer[0])
        print(rmsds[248][1], rmsds[248][2])
        file = os.path.join(pair_path, 'translated_conformer_248.mae')
        with structure.StructureWriter(file) as best_match:
            best_match.append(rmsds[248][0])

if __name__=="__main__":
    main()