"""
The purpose of this code is to first create the raw directory folder and include the following files
starting protein receptor
starting ligand
target ligand
glide pose viewer file

Then the top glide poses are added

Then the decoys are created

It can be run on sherlock using
$ $SCHRODINGER/run python3 systematic_decoy_search.py get_clash_data /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /home/users/sidhikab/lig_clash_score/src/data/decoy_timing_data_with_clash --rotation_search_step_size 5 --grid_size 1 --grid_n 1 --remove_prot_h --prot_pocket_only --clash_filter --clash_cutoff 15 --protein A5F5R2 --target 4x24 --start 4wkb --index 0
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
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

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
    for dx in range(-grid_size, grid_size + 1):
        for dy in range(-grid_size, grid_size + 1):
            for dz in range(-grid_size, grid_size + 1):
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


def get_pairs(process, raw_root, grid_size, include_clash_filter):
    pairs = []
    if include_clash_filter:
        print("get grid groups")
        grouped_data_files = get_grid_groups(6, 30)
        print("get grid data")
        data_dict = get_grid_data(process, grouped_data_files, raw_root, 6)
        print("create pairs")
        for pair in data_dict:
            if sum(data_dict[pair][:grid_size + 1]) != 0:
                pairs.append(pair)
    else:
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
                pairs.append((protein, target, start))
    return pairs


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


def time_conformer_decoys(pair_path, target_lig, prot):
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


def create_conformer_decoys(grid, target_lig, prot, cutoff, rotation_search_step_size, protein, target, start, index,
                            pair_path, data_folder, include_clash_filter, test, x_rot, y_rot, z_rot):
    data_dict = {'protein': [], 'target': [], 'start': [], 'num_conformers': [], 'num_poses_searched': [],
                 'num_correct_poses_found': [], 'time_elapsed': [], 'time_elapsed_per_conformer': [], 'grid_loc_x': [],
                 'grid_loc_y': [], 'grid_loc_z': []}
    clash_data = []

    for grid_loc in grid:
        counter = 0
        num_correct_found = 0
        conformer_file = os.path.join(pair_path, "aligned_to_start_without_hydrogen_conformers.mae")
        conformers = list(structure.StructureReader(conformer_file))
        decoy_start_time = time.time()

        for conformer in conformers:
            transform.translate_structure(conformer, grid_loc[0], grid_loc[1], grid_loc[2])
            conformer_center = list(get_centroid(conformer))
            coords = conformer.getXYZ(copy=True)

            for x in range(-30, 30 + rotation_search_step_size, rotation_search_step_size):
                for y in range(-30, 30 + rotation_search_step_size, rotation_search_step_size):
                    for z in range(-30, 30 + rotation_search_step_size, rotation_search_step_size):
                        counter += 1
                        new_coords = rotate_structure(coords, math.radians(x), math.radians(y),
                                                      math.radians(z), conformer_center)
                        conformer.setXYZ(new_coords)

                        if test and x_rot == x and y_rot == y and z_rot == z:
                            return conformer

                        rmsd_val = rmsd.calculate_in_place_rmsd(conformer, conformer.getAtomIndices(), target_lig,
                                                                target_lig.getAtomIndices())

                        if include_clash_filter:
                            clash = steric_clash.clash_volume(prot, struc2=conformer)
                            clash_data.append((clash, rmsd_val))
                        if rmsd_val < cutoff:
                            num_correct_found += 1

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

    df = pd.DataFrame.from_dict(data_dict)
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    save_folder = os.path.join(data_folder, '{}_{}-to-{}'.format(protein, target, start))
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    df.to_csv(os.path.join(save_folder, '{}.csv'.format(index)))
    if include_clash_filter:
        outfile = open(os.path.join(save_folder, 'clash_data_{}.pkl'.format(index)), 'wb')
        pickle.dump(clash_data, outfile)
    return None


def search_system_caller(pairs, raw_root, run_path, docked_prot_file, save_folder, rotation_search_step_size, grid_size,
                         grid_n, include_clash_filter, pocket_only, no_prot_h, grouped_files):
    for protein, target, start in pairs:
        for i in range(len(grouped_files)):
            cmd = 'sbatch -p owners -t 3:00:00 -o {} --wrap="$SCHRODINGER/run python3 systematic_decoy_search.py ' \
                  'search {} {} {} {} --protein {} --target {} --start {} --rotation_search_step_size {} ' \
                  '--grid_size {} --grid_n {} --index {}'
            if include_clash_filter:
                cmd += ' --clash_filter'
            if pocket_only:
                cmd += ' --prot_pocket_only'
            if no_prot_h:
                cmd += ' --remove_prot_h'
            cmd += '"'
            out_file_name = '{}_{}-to-{}_{}.out'.format(protein, target, start, i)
            os.system(cmd.format(os.path.join(run_path, out_file_name), docked_prot_file, run_path, raw_root,
                                 save_folder, protein, target, start, rotation_search_step_size, grid_size, grid_n, i))


def run_search(protein, target, start, index, raw_root, data_folder, get_time, cutoff, rotation_search_step_size, grid,
               no_prot_h, pocket_only, include_clash_filter, test=False, x_rot=0, y_rot=0, z_rot=0):
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
        conformer = create_conformer_decoys(grid, target_lig, prot, cutoff, rotation_search_step_size, protein, target,
                                            start, index, pair_path, data_folder, include_clash_filter, test, x_rot,
                                            y_rot, z_rot)
        return conformer


def run_test_search(protein, target, start, raw_root, cutoff, rotation_search_step_size, pair_path, no_prot_h,
                    pocket_only, get_time):
    angles = [i for i in range(-30, 30 + rotation_search_step_size, rotation_search_step_size)]
    angles = angles[:5]
    x_rot = random.choice(angles)
    y_rot = random.choice(angles)
    z_rot = random.choice(angles)
    grid_points = [i for i in range(-6, 7)]
    grid = [[random.choice(grid_points), random.choice(grid_points), random.choice(grid_points)]]

    conformer = run_search(protein, target, start, 0, raw_root, get_time, cutoff, rotation_search_step_size, grid,
                           no_prot_h, pocket_only, True, x_rot, y_rot, z_rot)

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


def get_grid_data(process, grouped_files, raw_root, grid_size):
    dfs = []
    pairs = []
    counter = 0
    print("going through process")
    for protein, target, start in process:
        print(protein, target, start)
        if counter == 10:
            break
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, pair)
        conformer_file = os.path.join(pair_path, "{}_lig0-out.maegz".format(target))
        conformers = list(structure.StructureReader(conformer_file))
        print("after read file")
        if len(conformers) == 1:
            continue
        else:
            counter += 1
        pairs.append((protein, target, start))
        save_folder = os.path.join(os.getcwd(), 'decoy_timing_data', '{}_{}-to-{}'.format(protein, target, start))
        for i in range(len(grouped_files)):
            print(i)
            dfs.append(pd.read_csv(os.path.join(save_folder, '{}.csv'.format(i))))

    print("concat")
    data_df = pd.concat(dfs)
    pair_dict = {}

    for pair in pairs:
        print(pair)
        protein, target, start = pair
        pair_df = data_df[
            (data_df['protein'] == protein) & (data_df['target'] == target) & (data_df['start'] == start)]
        pair_dict[pair] = []
        for i in range(grid_size + 1):
            grid_df = pair_df[(pair_df['grid_loc_x'] <= i) & (pair_df['grid_loc_x'] >= -i) &
                              (pair_df['grid_loc_y'] <= i) & (pair_df['grid_loc_y'] >= -i) &
                              (pair_df['grid_loc_z'] <= i) & (pair_df['grid_loc_z'] >= -i)]
            pair_dict[pair].append(grid_df.sum(axis=0)['num_correct_poses_found'])

    return pair_dict


def get_combined_data(process, grouped_files, raw_root, data_folder, grid_size, include_clash_filter):
    dfs = []
    pairs = get_pairs(process, raw_root, grid_size, include_clash_filter)
    for protein, target, start in pairs:
        save_folder = os.path.join(data_folder, '{}_{}-to-{}'.format(protein, target, start))
        for i in range(len(grouped_files)):
            dfs.append(pd.read_csv(os.path.join(save_folder, '{}.csv'.format(i))))

    data_df = pd.concat(dfs)
    data_df.to_csv(os.path.join(data_folder, 'combined_with_target_clash_filter.csv'))


def graph_clash(process, grouped_files, raw_root, data_folder, grid_size, rmsd_cutoff, include_clash_filter):
    clash_data = []
    pairs = get_pairs(process, raw_root, grid_size, include_clash_filter)
    for protein, target, start in tqdm(pairs, desc="going through protein-ligand pairs to get clash data"):
        save_folder = os.path.join(data_folder, '{}_{}-to-{}'.format(protein, target, start))
        for i in range(len(grouped_files)):
            infile = open(os.path.join(save_folder, 'clash_data_{}.pkl'.format(i)), 'rb')
            data = pickle.load(infile)
            infile.close()
            clash_data.extend(data)

    # print("separating data")
    # correct_x = [x[1] for x in clash_data if x[1] < rmsd_cutoff]
    # correct_y = [x[0] for x in clash_data if x[1] < rmsd_cutoff]
    # incorrect = [x for x in clash_data if x[1] >= rmsd_cutoff]
    # random.shuffle(incorrect)
    # incorrect = incorrect[:2000000]
    # incorrect_x = [x[1] for x in incorrect]
    # incorrect_y = [x[0] for x in incorrect]
    #
    # fig, ax = plt.subplots()
    # print("graphing correct")
    # plt.scatter(correct_x, correct_y, label='RMSD < 2 A')
    # print("graphing incorrect")
    # plt.scatter(incorrect_x, incorrect_y, label='RMSD >= 2 A')
    # print("finished graphing")
    # ax.legend()
    # ax.set_xlabel('RMSD')
    # ax.set_ylabel('Clash Volume')
    # plt.savefig(os.path.join(data_folder, 'rmsd_vs_clash.png'))

    correct = [x[0] for x in clash_data if x[1] < rmsd_cutoff and x[0] < 200]
    incorrect = [x[0] for x in clash_data if x[1] >= rmsd_cutoff]
    random.shuffle(incorrect)
    incorrect = incorrect[:2000000]
    incorrect = [x for x in incorrect if x < 200]

    fig, ax = plt.subplots()
    print("graphing correct")
    sns.distplot(correct, hist=True, label='RMSD < 2 A');
    print("graphing incorrect")
    sns.distplot(incorrect, hist=True, label='RMSD >= 2 A');
    print("finished graphing")
    ax.legend()
    ax.set_xlabel('Clash Volume')
    ax.set_ylabel('Frequency')
    plt.savefig(os.path.join(data_folder, 'target_clash_frequency_sub_200.png'))


def get_clash_data(process, grouped_files, raw_root, data_folder, grid_size, rmsd_cutoff, clash_cutoff,
                   include_clash_filter):
    # pairs = get_pairs(process, raw_root, grid_size, include_clash_filter)
    # for protein, target, start in tqdm(pairs, desc="going through protein-ligand pairs to get clash data"):
    #     save_folder = os.path.join(data_folder, '{}_{}-to-{}'.format(protein, target, start))
    #     for i in range(len(grouped_files)):
    #         infile = open(os.path.join(save_folder, 'clash_data_{}.pkl'.format(i)), 'rb')
    #         data = pickle.load(infile)
    #         infile.close()
    #
    #         num_after_clash_filter = len([x for x in data if x[0] < clash_cutoff])
    #         num_correct_after_clash_filter = len([x for x in data if x[0] < clash_cutoff and x[1] < rmsd_cutoff])
    #
    #         data_file = os.path.join(save_folder, '{}.csv'.format(i))
    #         df = pd.read_csv(data_file)
    #         df = df.rename(columns={'num_after_clash_filter': 'num_after_clash_filter_cutoff_200',
    #                                     'num_correct_after_clash_filter': 'num_correct_after_clash_filter_cutoff_200'})
    #         df['num_after_clash_filter_cutoff_{}'.format(clash_cutoff)] = [num_after_clash_filter]
    #         df['num_correct_after_clash_filter_cutoff_{}'.format(clash_cutoff)] = [num_correct_after_clash_filter]
    #         df.to_csv(data_file)

    print("get pairs")
    pairs = get_pairs(process, raw_root, grid_size, include_clash_filter)
    for protein, target, start in pairs:
        print(protein, target, start)
        out_folder = os.path.join('decoy_timing_data_with_clash', '{}_{}-to-{}'.format(protein, target, start))
        in_folder = os.path.join('decoy_timing_data_with_target_clash', '{}_{}-to-{}'.format(protein, target, start))
        print(out_folder, in_folder)
        for i in range(len(grouped_files)):
            print(i)
            infile = open(os.path.join(in_folder, 'clash_data_{}.pkl'.format(i)), 'rb')
            target_clash_data = pickle.load(infile)
            infile.close()

            infile = open(os.path.join(out_folder, 'clash_data_{}.pkl'.format(i)), 'rb')
            start_clash_data = pickle.load(infile)
            infile.close()

            num_after_clash_filter = 0
            num_correct_after_clash_filter = 0

            for j in range(len(target_clash_data)):
                if target_clash_data[j][0] < 15 and start_clash_data[j][0] < 100:
                    num_after_clash_filter += 1
                    if target_clash_data[j][1] < rmsd_cutoff:
                        num_correct_after_clash_filter += 1

            data_file = os.path.join(out_folder, '{}.csv'.format(i))
            df = pd.read_csv(data_file)
            df['num_after_simple_100_ideal_advanced_15'] = [num_after_clash_filter]
            df['num_correct_after_simple_100_ideal_advanced_15'] = [num_correct_after_clash_filter]
            df.to_csv(data_file)
            return


def analyze_clash_data(process, raw_root, data_folder, grid_size, include_clash_filter):
    df = pd.read_csv(os.path.join(data_folder, 'combined_with_target_clash_filter.csv'))
    pairs = get_pairs(process, raw_root, grid_size, include_clash_filter)
    for protein, target, start in pairs:
        print(protein, target, start)
        for cutoff in [100, 200]:
            print('\tCUTOFF =', cutoff)
            pair_df = df[(df['protein'] == protein) & (df['target'] == target) & (df['start'] == start)]
            fraction_before_filter = sum(pair_df['num_correct_poses_found']) / sum(pair_df['num_poses_searched'])
            fraction_after_filter = sum(pair_df['num_correct_after_clash_filter_cutoff_{}'.format(cutoff)]) / \
                                    sum(pair_df['num_after_clash_filter_cutoff_{}'.format(cutoff)])

            print('\t\tfraction_before_filter =', fraction_before_filter, ', fraction_after_filter =',
                  fraction_after_filter)

            num_poses_at_start = sum(pair_df['num_poses_searched'])
            num_poses_after_clash_filter = sum(pair_df['num_after_clash_filter_cutoff_{}'.format(cutoff)])
            num_cut = num_poses_at_start - num_poses_after_clash_filter
            print('\t\tnum_poses_at_start =', num_poses_at_start, ', num_poses_after_filter =', num_poses_after_clash_filter,
                  ', num_poses_cut =', num_cut, ', fraction_cut =', num_cut / num_poses_at_start)

            num_correct_at_start = sum(pair_df['num_correct_poses_found'])
            num_correct_after_clash_filter = sum(pair_df['num_correct_after_clash_filter_cutoff_{}'.format(cutoff)])
            num_cut = num_correct_at_start - num_correct_after_clash_filter
            print('\t\tnum_correct_at_start =', num_correct_at_start, ', num_correct_after_filter =',
                  num_correct_after_clash_filter, ', num_poses_cut =', num_cut, ', fraction_cut =',
                  num_cut / num_correct_at_start)

        cutoff = 15
        print('\tTARGET CLASH CUTOFF =', cutoff)
        pair_df = df[(df['protein'] == protein) & (df['target'] == target) & (df['start'] == start)]
        fraction_before_filter = sum(pair_df['num_correct_poses_found']) / sum(pair_df['num_poses_searched'])
        fraction_after_filter = sum(pair_df['num_correct_after_target_clash_filter_cutoff_{}'.format(cutoff)]) / \
                                sum(pair_df['num_after_target_clash_filter_cutoff_{}'.format(cutoff)])

        print('\t\tfraction_before_filter =', fraction_before_filter, ', fraction_after_filter =',
              fraction_after_filter)

        num_poses_at_start = sum(pair_df['num_poses_searched'])
        num_poses_after_clash_filter = sum(pair_df['num_after_target_clash_filter_cutoff_{}'.format(cutoff)])
        num_cut = num_poses_at_start - num_poses_after_clash_filter
        print('\t\tnum_poses_at_start =', num_poses_at_start, ', num_poses_after_filter =',
              num_poses_after_clash_filter,
              ', num_poses_cut =', num_cut, ', fraction_cut =', num_cut / num_poses_at_start)

        num_correct_at_start = sum(pair_df['num_correct_poses_found'])
        num_correct_after_clash_filter = sum(pair_df['num_correct_after_target_clash_filter_cutoff_{}'.format(cutoff)])
        num_cut = num_correct_at_start - num_correct_after_clash_filter
        print('\t\tnum_correct_at_start =', num_correct_at_start, ', num_correct_after_filter =',
              num_correct_after_clash_filter, ', num_poses_cut =', num_cut, ', fraction_cut =',
              num_cut / num_correct_at_start)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either align or search')
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('save_folder', type=str, help='directory where data will be saved')
    parser.add_argument('--protein', type=str, default='', help='protein name')
    parser.add_argument('--target', type=str, default='', help='target ligand name')
    parser.add_argument('--start', type=str, default='', help='start ligand name')
    parser.add_argument('--align_n', type=int, default=10, help='number of alignments processed in each job')
    parser.add_argument('--rotation_search_step_size', type=int, default=1, help='step size between each angle '
                                                                                 'checked, in degrees')
    parser.add_argument('--index', type=int, default=-1, help='grid point group index')
    parser.add_argument('--rmsd_cutoff', type=int, default=2, help='rmsd accuracy cutoff between predicted ligand pose '
                                                                   'and true ligand pose')
    parser.add_argument('--clash_cutoff', type=int, default=100, help='clash volume cutoff for filter')
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
    parser.add_argument('--clash_filter', dest='include_clash_filter', action='store_true')
    parser.add_argument('--no_clash_filter', dest='include_clash_filter', action='store_false')
    parser.set_defaults(include_clash_filter=False)

    args = parser.parse_args()

    random.seed(0)

    if not os.path.exists(args.run_path):
        os.mkdir(args.run_path)

    elif args.task == 'get_target_clash':
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
            prot_file = os.path.join(pair_path, '{}_prot.mae'.format(target))
            prot = list(structure.StructureReader(prot_file))[0]
            target_lig_file = os.path.join(pair_path, 'ligand_poses', '{}_lig0.mae'.format(target))
            target_lig = list(structure.StructureReader(target_lig_file))[0]
            clash = steric_clash.clash_volume(prot, struc2=target_lig)
            print(protein, target, start, clash)


if __name__=="__main__":
    main()