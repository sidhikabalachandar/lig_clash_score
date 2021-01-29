"""
The purpose of this code is to create conformers

It can be run on sherlock using
$ $SCHRODINGER/run python3 search.py all_search /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /home/users/sidhikab/lig_clash_score/src/data/decoy_timing_data_with_clash --grid_size 1 --n 1
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
from tqdm import tqdm
import pickle
import scipy.spatial
import time
import math
import numpy as np
import statistics
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
        for line in tqdm(fp, desc='index file'):
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

# MAIN TASK FUNCTIONS


def all_search(pairs, raw_root, run_path, docked_prot_file, save_folder, rotation_search_step_size, grid_size, n,
               target_clash_cutoff, include_clash_filter, pocket_only, no_prot_h, save_poses, grouped_files):
    for protein, target, start in pairs:
        for i in range(len(grouped_files)):
            cmd = 'sbatch -p rondror -t 6:00:00 -o {} --wrap="$SCHRODINGER/run python3 search.py search {} {} {} {} ' \
                  '--protein {} --target {} --start {} --rotation_search_step_size {} --grid_size {} --n {} ' \
                  '--target_clash_cutoff {} --index {}'
            if not include_clash_filter:
                cmd += ' --no_clash_filter'
            if not pocket_only:
                cmd += ' --all_prot'
            if not no_prot_h:
                cmd += ' --keep_prot_h'
            if save_poses:
                cmd += ' --save'
            cmd += '"'
            out_file_name = '{}_{}-to-{}_{}.out'.format(protein, target, start, i)
            os.system(cmd.format(os.path.join(run_path, out_file_name), docked_prot_file, run_path, raw_root,
                                 save_folder, protein, target, start, rotation_search_step_size, grid_size, n,
                                 target_clash_cutoff, i))


def search(protein, target, start, index, raw_root, save_folder, get_time, rmsd_cutoff, start_clash_cutoff,
           target_clash_cutoff, rotation_search_step_size, grid, no_prot_h, pocket_only, include_clash_filter,
           save_poses, test=False, x_rot=0, y_rot=0, z_rot=0):
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
                                 target_clash_cutoff, rotation_search_step_size, protein, target, start, index,
                                 pair_path, save_folder, include_clash_filter, save_poses, test, x_rot, y_rot, z_rot)

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


def create_poses(grid, target_lig, start_prot, target_prot, rmsd_cutoff, start_clash_cutoff, target_clash_cutoff,
                 rotation_search_step_size, protein, target, start, index, pair_path, save_folder, include_clash_filter,
                 save_poses, test, x_rot, y_rot, z_rot):
    data_dict = {'protein': [], 'target': [], 'start': [], 'num_conformers': [], 'num_poses_searched': [],
                 'num_correct_poses_found': [], 'time_elapsed': [], 'time_elapsed_per_conformer': [], 'grid_loc_x': [],
                 'grid_loc_y': [], 'grid_loc_z': []}
    clash_data = []
    pose_path = os.path.join(pair_path, 'grid_search_poses')
    if not os.path.exists(pose_path):
        os.mkdir(pose_path)

    saved_dict = {'name': [], 'start_clash': [], 'target_clash': []}

    for grid_loc in grid:
        counter = 0
        num_correct_found = 0
        conformer_file_without_hydrogen = os.path.join(pair_path, "aligned_to_start_without_hydrogen_conformers.mae")
        conformer_file_with_hydrogen = os.path.join(pair_path, "aligned_to_start_with_hydrogen_conformers.mae")
        conformers_without_hydrogen = list(structure.StructureReader(conformer_file_without_hydrogen))
        conformers_with_hydrogen = list(structure.StructureReader(conformer_file_with_hydrogen))
        pose_file = os.path.join(pose_path, '{}_{}_{}.maegz'.format(grid_loc[0], grid_loc[1], grid_loc[2]))

        with structure.StructureWriter(pose_file) as pose:
            decoy_start_time = time.time()
            for i, conformer_without_hydrogen in enumerate(conformers_without_hydrogen):
                translate_structure(conformer_without_hydrogen, grid_loc[0], grid_loc[1], grid_loc[2])
                conformer_center_without_hydrogen = list(get_centroid(conformer_without_hydrogen))
                coords_without_hydrogen = conformer_without_hydrogen.getXYZ(copy=True)

                conformer_with_hydrogen = conformers_with_hydrogen[i]
                translate_structure(conformer_with_hydrogen, grid_loc[0], grid_loc[1], grid_loc[2])
                conformer_center_with_hydrogen = list(get_centroid(conformer_with_hydrogen))
                coords_with_hydrogen = conformer_with_hydrogen.getXYZ(copy=True)

                for x in range(-30, 30 + rotation_search_step_size, rotation_search_step_size):
                    for y in range(-30, 30 + rotation_search_step_size, rotation_search_step_size):
                        for z in range(-30, 30 + rotation_search_step_size, rotation_search_step_size):
                            counter += 1
                            new_coords = rotate_structure(coords_without_hydrogen, math.radians(x), math.radians(y),
                                                          math.radians(z), conformer_center_without_hydrogen)
                            conformer_without_hydrogen.setXYZ(new_coords)

                            if test and x_rot == x and y_rot == y and z_rot == z:
                                return conformer_without_hydrogen

                            if save_poses:
                                start_clash = steric_clash.clash_volume(start_prot, struc2=conformer_without_hydrogen)
                                if start_clash < start_clash_cutoff:
                                    target_clash = steric_clash.clash_volume(target_prot,
                                                                             struc2=conformer_without_hydrogen)
                                    if target_clash < target_clash_cutoff:
                                        name = '{}_{},{},{}_{},{},{}'.format(i, grid_loc[0], grid_loc[1],
                                                                                           grid_loc[2], x, y, z)
                                        new_coords = rotate_structure(coords_with_hydrogen, math.radians(x),
                                                                      math.radians(y), math.radians(z),
                                                                      conformer_center_with_hydrogen)
                                        conformer_with_hydrogen.setXYZ(new_coords)

                                        conformer_with_hydrogen.title = name
                                        pose.append(conformer_with_hydrogen)
                                        saved_dict['name'].append(name)
                                        saved_dict['start_clash'].append(start_clash)
                                        saved_dict['target_clash'].append(target_clash)

                            else:
                                rmsd_val = rmsd.calculate_in_place_rmsd(conformer_without_hydrogen,
                                                                        conformer_without_hydrogen.getAtomIndices(),
                                                                        target_lig, target_lig.getAtomIndices())
                                if include_clash_filter:
                                    clash = steric_clash.clash_volume(start_prot, struc2=conformer_without_hydrogen)
                                    clash_data.append((clash, rmsd_val))

                                if rmsd_val < rmsd_cutoff:
                                    num_correct_found += 1

            decoy_end_time = time.time()

        if save_poses:
            df = pd.DataFrame.from_dict(saved_dict)
            df.to_csv(os.path.join(pose_path, '{}_{}_{}.csv'.format(grid_loc[0], grid_loc[1], grid_loc[2])))
        else:
            data_dict['protein'].append(protein)
            data_dict['target'].append(target)
            data_dict['start'].append(start)
            data_dict['num_conformers'].append(len(conformers_without_hydrogen))
            data_dict['num_poses_searched'].append(counter)
            data_dict['num_correct_poses_found'].append(num_correct_found)
            data_dict['time_elapsed'].append(decoy_end_time - decoy_start_time)
            data_dict['time_elapsed_per_conformer'].append((decoy_end_time - decoy_start_time) /
                                                           len(conformers_without_hydrogen))
            data_dict['grid_loc_x'].append(grid_loc[0])
            data_dict['grid_loc_y'].append(grid_loc[1])
            data_dict['grid_loc_z'].append(grid_loc[2])

    if not save_poses:
        df = pd.DataFrame.from_dict(data_dict)
        pair_folder = os.path.join(save_folder, '{}_{}-to-{}'.format(protein, target, start))
        if not os.path.exists(pair_folder):
            os.mkdir(pair_folder)
        df.to_csv(os.path.join(pair_folder, '{}.csv'.format(index)))
        if include_clash_filter:
            outfile = open(os.path.join(pair_folder, 'clash_data_{}.pkl'.format(index)), 'wb')
            pickle.dump(clash_data, outfile)


def check_search(pairs, raw_root):
    for protein, target, start in pairs:
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, pair)
        pose_path = os.path.join(pair_path, 'grid_search_poses')
        csv_counter = 0
        pose_counter = 0

        for file in os.listdir(pose_path):
            if file[-3:] == 'csv':
                df = pd.read_csv(os.path.join(pose_path, file))
                csv_counter += len(df)
            elif file[-5:] == 'maegz':
                grid_poses = list(structure.StructureReader(os.path.join(pose_path, file)))
                pose_counter += len(grid_poses)

        print(protein, target, start)
        print("Num poses created:", pose_counter)
        print("Num poses in csv:", csv_counter)


def combine(pairs, grouped_files, save_folder):
    dfs = []
    for protein, target, start in pairs:
        pair_folder = os.path.join(save_folder, '{}_{}-to-{}'.format(protein, target, start))
        for i in range(len(grouped_files)):
            dfs.append(pd.read_csv(os.path.join(pair_folder, '{}.csv'.format(i))))

    data_df = pd.concat(dfs)
    data_df.to_csv(os.path.join(save_folder, 'combined_with_stacked_filter.csv'))


def grid_data(pairs, grouped_files, grid_size, save_folder, grid_file):
    pair_dict = {}

    for pair in pairs:
        protein, target, start = pair
        pair_dict[pair] = []
        pair_folder = os.path.join(save_folder, '{}_{}-to-{}'.format(protein, target, start))
        dfs = []
        for i in range(len(grouped_files)):
            dfs.append(pd.read_csv(os.path.join(pair_folder, '{}.csv'.format(i))))

        df = pd.concat(dfs)
        for grid_i in range(grid_size + 1):
            grid_df = df[(df['grid_loc_x'] <= grid_i) & (df['grid_loc_x'] >= -grid_i) &
                         (df['grid_loc_y'] <= grid_i) & (df['grid_loc_y'] >= -grid_i) &
                         (df['grid_loc_z'] <= grid_i) & (df['grid_loc_z'] >= -grid_i)]
            pair_dict[pair].append(grid_df.sum(axis=0)['num_correct_poses_found'])

    outfile = open(grid_file, 'wb')
    pickle.dump(pair_dict, outfile)
    print(pair_dict)



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
                pocket_only, get_time, include_clash_filter):
    angles = [i for i in range(-30, 30 + rotation_search_step_size, rotation_search_step_size)]
    angles = angles[:5]
    x_rot = random.choice(angles)
    y_rot = random.choice(angles)
    z_rot = random.choice(angles)
    grid_points = [i for i in range(-6, 7)]
    grid = [[random.choice(grid_points), random.choice(grid_points), random.choice(grid_points)]]

    conformer = search(protein, target, start, 0, raw_root, save_folder, get_time, cutoff, rotation_search_step_size,
                       grid, no_prot_h, pocket_only, include_clash_filter, True, x_rot, y_rot, z_rot)

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
    parser.add_argument('--num_conformers', type=int, default=300, help='maximum number of conformers considered')
    parser.add_argument('--num_pairs', type=int, default=10, help='number of protein-ligand pairs considered')
    parser.add_argument('--n', type=int, default=30, help='number of grid_points processed in each job')
    parser.add_argument('--grid_size', type=int, default=6, help='grid size in positive and negative x, y, z '
                                                                 'directions')
    parser.add_argument('--grid_file', type=str, default='', help='pickle file with grid data dictionary')
    parser.add_argument('--index', type=int, default=-1, help='grid point group index')
    parser.add_argument('--rotation_search_step_size', type=int, default=5, help='step size between each angle '
                                                                                 'checked, in degrees')
    parser.add_argument('--rmsd_cutoff', type=int, default=2, help='rmsd accuracy cutoff between predicted ligand pose '
                                                                   'and true ligand pose')
    parser.add_argument('--start_clash_cutoff', type=int, default=100, help='clash cutoff between start protein and '
                                                                            'ligand pose')
    parser.add_argument('--target_clash_cutoff', type=int, default=15, help='clash cutoff between target protein and '
                                                                            'ligand pose')
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
    parser.add_argument('--save', dest='save_poses', action='store_true')
    parser.add_argument('--no_save', dest='save_poses', action='store_false')
    parser.set_defaults(save_poses=False)
    args = parser.parse_args()

    random.seed(0)

    if not os.path.exists(args.run_path):
        os.mkdir(args.run_path)

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    if args.task == 'all_search':
        grouped_files = group_grid(args.n, args.grid_size)
        pairs = [('A5HZZ9', '3c8a', '3qw5'), ('O38732', '2i0a', '2q5k'), ('P07900', '2qg2', '1yet'),
                 ('P51449', '5vb6', '5ufr')]
        # if args.include_clash_filter:
        #     pairs = get_grid_prots(args.grid_file, args.grid_size)
        # else:
        #     process = get_prots(args.docked_prot_file)
        #     random.shuffle(process)
        #     pairs = get_conformer_prots(process, args.raw_root, args.num_pairs)
        all_search(pairs, args.raw_root, args.run_path, args.docked_prot_file, args.save_folder,
                   args.rotation_search_step_size, args.grid_size, args.n, args.target_clash_cutoff,
                   args.include_clash_filter, args.pocket_only, args.no_prot_h, args.save_poses, grouped_files)

    elif args.task == 'search':
        grouped_files = group_grid(args.n, args.grid_size)
        search(args.protein, args.target, args.start, args.index, args.raw_root, args.save_folder, args.get_time,
               args.rmsd_cutoff, args.start_clash_cutoff, args.target_clash_cutoff, args.rotation_search_step_size,
               grouped_files[args.index], args.no_prot_h, args.pocket_only, args.include_clash_filter, args.save_poses)

    elif args.task == 'check_search':
        # if args.include_clash_filter:
        #     pairs = get_grid_prots(args.grid_file, args.grid_size)
        # else:
        #     process = get_prots(args.docked_prot_file)
        #     random.shuffle(process)
        #     pairs = get_conformer_prots(process, args.raw_root, args.num_pairs)
        pairs = [('P18031', '1g7g', '1c83')]
        check_search(pairs, args.raw_root)

    elif args.task == 'combine':
        grouped_files = group_grid(args.n, args.grid_size)
        if args.include_clash_filter:
            pairs = get_grid_prots(args.grid_file, args.grid_size)
        else:
            process = get_prots(args.docked_prot_file)
            random.shuffle(process)
            pairs = get_conformer_prots(process, args.raw_root, args.num_pairs)
        combine(pairs, grouped_files, args.save_folder)

    elif args.task == 'grid_data':
        process = get_prots(args.docked_prot_file)
        random.shuffle(process)
        grouped_files = group_grid(args.n, args.grid_size)
        pairs = get_conformer_prots(process, args.raw_root, args.num_pairs)
        grid_data(pairs, grouped_files, args.grid_size, args.save_folder, args.grid_file)

    elif args.task == 'test_rotate_translate':
        test_rotate_translate(args.protein, args.target, args.start, args.raw_root)

    elif args.task == 'test_search':
        test_search(args.protein, args.target, args.start, args.raw_root, args.save_folder, args.cutoff,
                    args.rotation_search_step_size, args.no_prot_h, args.pocket_only, args.get_time,
                    args.include_clash_filter)

    elif args.task == 'stats':
        protein = 'P18031'
        target = '1g7g'
        start = '1c83'

        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(args.raw_root, protein)
        pair_path = os.path.join(protein_path, pair)
        # pose_path = os.path.join(pair_path, args.decoy_type)
        #
        # df = pd.read_csv(os.path.join(pose_path, 'combined_data.csv'))
        # print(len(df))
        # print(len(df[df['rmsd'] < 3]) / len(df))
        # print(min(df[df['rmsd'] < 3]))

        prot_file = os.path.join(pair_path, '{}_prot.mae'.format(target))
        prot = list(structure.StructureReader(prot_file))[0]
        lig_file = os.path.join(pair_path, 'ligand_poses', '{}_lig0.mae'.format(target))
        lig = list(structure.StructureReader(lig_file))[0]
        print(steric_clash.clash_volume(prot, struc2=lig))


if __name__ == "__main__":
    main()
