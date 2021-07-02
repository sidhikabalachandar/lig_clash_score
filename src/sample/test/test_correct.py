"""
The purpose of this code is to first create the raw directory folder and include the following files
starting protein receptor
starting ligand
target ligand
glide pose viewer file

Then the top glide poses are added

Then the decoys are created

It can be run on sherlock using
$ $SCHRODINGER/run python3 test_correct.py group /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits/search_test_incorrect_glide_index.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d --protein P00797 --target 3own --start 3d91 --index 0
"""

import argparse
import os
import random
import pandas as pd
import numpy as np
import math
import pickle
import statistics
from tqdm import tqdm
import schrodinger.structure as structure
from schrodinger.structutils.transform import get_centroid
import schrodinger.structutils.interactions.steric_clash as steric_clash
import time
import scipy.spatial

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


'''
This function gets the mean and standard deviation of all of the bfactors of a structure
:param s: the protein structure 
:return: the mean and the standard deviation of the list of bfactors associated with the protein structure
'''
def bfactor_stats(s):
    bfactors = []
    for m in list(s.molecule):
        for r in list(m.residue):
            bfactors.append(r.temperature_factor)
    return statistics.mean(bfactors), statistics.stdev(bfactors)


'''
This function finds the normalized bfactor for a particular residue
:param residues: a list of all residues in the protein structure
:param index: the index of the particular residue in question
:param avg: the average bfactor over all residues in the protein structure
:param sdev: the standard deviation calculated over all residues in the protein structure
:return: the normalized bfactor value
'''
def normalizedBFactor(r, avg, sdev):
    return (r.temperature_factor - avg) / sdev

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


def get_dim(s):
    at = s.getXYZ(copy=True)
    at = at * 2
    at = (np.around(at - 0.5)).astype(np.int16)
    origin = np.full((3), np.amin(at))
    at = at - origin
    dim = np.amax(at) + 1
    return dim, origin

def get_grid(s, dim, origin):
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
    at = at - origin
    s_grid = np.zeros((dim, dim, dim))
    np.add.at(s_grid, (at[:, 0], at[:, 1], at[:, 2]), 1)
    grids = []

    for m in list(s.molecule):
        for r in list(m.residue):
            r_at = r.extractStructure().getXYZ(copy=True)
            r_at = r_at * 2
            r_at = (np.around(r_at - 0.5)).astype(np.int16)
            r_at = r_at - origin
            grid = np.zeros((dim, dim, dim))
            np.add.at(grid, (r_at[:, 0], r_at[:, 1], r_at[:, 2]), 1)
            grids.append((grid, r))

    return s_grid, grids


def get_clash(s, grid, origin):
    at = s.getXYZ(copy=True)
    at = at * 2
    at = (np.around(at - 0.5)).astype(np.int16)
    at = at - origin
    return np.sum(grid[at[:, 0], at[:, 1], at[:, 2]])

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either align or search')
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('root', type=str, help='directory where raw data will be placed')
    parser.add_argument('--index', type=int, default=-1, help='index of pose file')
    parser.add_argument('--protein', type=str, default='', help='name of protein')
    parser.add_argument('--target', type=str, default='', help='name of target ligand')
    parser.add_argument('--start', type=str, default='', help='name of start ligand')
    parser.add_argument('--n', type=int, default=600, help='number of poses to process in each job')
    parser.add_argument('--save_pred_path', type=str, help='prediction graph file')
    parser.add_argument('--save_true_path', type=str, help='true graph file')
    parser.add_argument('--target_clash_cutoff', type=int, default=0, help='clash cutoff between target protein and '
                                                                            'ligand pose')
    parser.add_argument('--intolerable_cutoff', type=int, default=0, help='cutoff of max num intolerable residues')

    args = parser.parse_args()
    random.seed(0)

    raw_root = os.path.join(args.root, 'raw')

    if not os.path.exists(args.run_path):
        os.mkdir(args.run_path)

    if args.task == 'all':
        pairs = get_prots(args.docked_prot_file)
        random.shuffle(pairs)
        counter = 0
        for protein, target, start in pairs[:5]:
            if protein == 'Q86WV6':
                continue
            pair = '{}-to-{}'.format(target, start)
            protein_path = os.path.join(raw_root, protein)
            pair_path = os.path.join(protein_path, pair)
            grid_size = get_grid_size(pair_path, target, start)
            pose_path = os.path.join(pair_path, 'exhaustive_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size))
            correct_path = os.path.join(pose_path, 'correct_after_simple_filter')
            file = os.path.join(correct_path, 'combined.csv')
            df = pd.read_csv(file)
            indices = [i for i in range(len(df))]
            grouped_indices = group_files(args.n, indices)

            for i in range(len(grouped_indices)):
                cmd = 'sbatch -p rondror -t 0:20:00 -o {} --wrap="$SCHRODINGER/run python3 test_correct.py group {} {} ' \
                      '{} --protein {} --target {} --start {} --index {}"'
                counter += 1
                os.system(cmd.format(os.path.join(args.run_path, 'test_{}_{}_{}.out'.format(protein, pair, i)),
                                     args.docked_prot_file, args.run_path, args.root, protein, target, start, i))

        print(counter)

    elif args.task == "group":
        pair = '{}-to-{}'.format(args.target, args.start)
        protein_path = os.path.join(raw_root, args.protein)
        pair_path = os.path.join(protein_path, pair)
        grid_size = get_grid_size(pair_path, args.target, args.start)
        pose_path = os.path.join(pair_path, 'exhaustive_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size))
        correct_path = os.path.join(pose_path, 'correct_after_simple_filter')
        file = os.path.join(correct_path, 'combined.csv')
        df = pd.read_csv(file)
        all_indices = [i for i in range(len(df))]
        grouped_indices = group_files(args.n, all_indices)

        clash_path = os.path.join(correct_path, 'clash_data')
        if not os.path.exists(clash_path):
            os.mkdir(clash_path)
        out_clash_file = os.path.join(clash_path, 'clash_data_{}.csv'.format(args.index))

        # if not os.path.exists(out_clash_file):
        prot_docking = list(structure.StructureReader(os.path.join(pair_path, '{}_prot.mae'.format(args.start))))[0]
        conformer_file = os.path.join(pair_path, "aligned_to_start_with_hydrogen_conformers.mae")
        conformers = list(structure.StructureReader(conformer_file))
        mean, stdev = bfactor_stats(prot_docking)

        with open('{}/{}_mcss.csv'.format(pair_path, pair)) as f:
            mcss = int(f.readline().strip().split(',')[4])

        docking_dim, docking_origin = get_dim(prot_docking)

        # pocket res only
        target_lig_file = os.path.join(pair_path, 'ligand_poses', '{}_lig0.mae'.format(args.target))
        target_lig = list(structure.StructureReader(target_lig_file))[0]
        get_pocket_res(prot_docking, target_lig, 6)

        # clash preprocessing
        prot_docking_grid, grids = get_grid(prot_docking, docking_dim, docking_origin)

        # create feature dictionary
        clash_features = {'name': [], 'residue': [], 'bfactor': [], 'mcss': [],
                          'volume_docking': []}

        for i in grouped_indices[args.index]:
            name = df.loc[[i]]['name'].iloc[0]
            conformer_index = df.loc[[i]]['conformer_index'].iloc[0]
            c = conformers[conformer_index]
            old_coords = c.getXYZ(copy=True)
            grid_loc_x = df.loc[[i]]['grid_loc_x'].iloc[0]
            grid_loc_y = df.loc[[i]]['grid_loc_y'].iloc[0]
            grid_loc_z = df.loc[[i]]['grid_loc_z'].iloc[0]
            translate_structure(c, grid_loc_x, grid_loc_y, grid_loc_z)
            conformer_center = list(get_centroid(c))
            coords = c.getXYZ(copy=True)
            rot_x = df.loc[[i]]['rot_x'].iloc[0]
            rot_y = df.loc[[i]]['rot_y'].iloc[0]
            rot_z = df.loc[[i]]['rot_z'].iloc[0]
            new_coords = rotate_structure(coords, math.radians(rot_x), math.radians(rot_y), math.radians(rot_z),
                                          conformer_center)

            # for clash features dictionary
            c.setXYZ(new_coords)

            for grid, r in grids:
                # res_clash = get_clash(c, grid, docking_origin)
                volume_docking = steric_clash.clash_volume(prot_docking, atoms1=r.getAtomIndices(), struc2=c)
                if volume_docking != 0:
                    clash_features['name'].append(name)
                    clash_features['residue'].append(r.getAsl())
                    clash_features['bfactor'].append(normalizedBFactor(r, mean, stdev))
                    clash_features['mcss'].append(mcss)
                    clash_features['volume_docking'].append(volume_docking)

            c.setXYZ(old_coords)

        out_clash_df = pd.DataFrame.from_dict(clash_features)
        infile = open(os.path.join(args.root, 'clash_classifier.pkl'), 'rb')
        clf = pickle.load(infile)
        infile.close()
        if len(out_clash_df) != 0:
            pred = clf.predict(out_clash_df[['bfactor', 'mcss', 'volume_docking']])
        else:
            pred = []
        print(sum(pred))
        out_clash_df['pred'] = pred
        out_clash_df.to_csv(out_clash_file, index=False)

        # add res info to pose data frame
        zeros = [0 for _ in range(len(df))]
        df['pred_num_intolerable'] = zeros
        df['num_clash_docking'] = zeros

        for i in grouped_indices[args.index]:
            name = df.loc[[i]]['name'].iloc[0]
            name_df = out_clash_df[out_clash_df['name'] == name]
            if len(name_df) != 0:
                pred = name_df['pred'].to_list()
                print(sum(pred))
                df.iat[i, df.columns.get_loc('pred_num_intolerable')] = sum(pred)
                df.iat[i, df.columns.get_loc('num_clash_docking')] = len(pred)

            # else
            # default set to 0

        df = df.iloc[grouped_indices[args.index], :]
        df.to_csv(os.path.join(clash_path, 'pose_pred_data_{}.csv'.format(args.index)))

    elif args.task == 'check':
        pairs = get_prots(args.docked_prot_file)
        random.shuffle(pairs)
        missing = []
        counter = 0
        for protein, target, start in pairs[:5]:
            if protein == 'Q86WV6':
                continue
            pair = '{}-to-{}'.format(target, start)
            protein_path = os.path.join(raw_root, protein)
            pair_path = os.path.join(protein_path, pair)
            grid_size = get_grid_size(pair_path, target, start)
            pose_path = os.path.join(pair_path, 'exhaustive_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size))
            correct_path = os.path.join(pose_path, 'correct_after_simple_filter')
            clash_path = os.path.join(correct_path, 'clash_data')
            file = os.path.join(correct_path, 'combined.csv')
            df = pd.read_csv(file)
            indices = [i for i in range(len(df))]
            grouped_indices = group_files(args.n, indices)

            for i in range(len(grouped_indices)):
                out_clash_file = os.path.join(clash_path, 'clash_data_{}.csv'.format(i))
                pose_file = os.path.join(clash_path, 'pose_pred_data_{}.csv'.format(i))
                counter += 1
                if not os.path.exists(out_clash_file):
                    missing.append((protein, target, start, i))
                    continue
                if not os.path.exists(pose_file):
                    missing.append((protein, target, start, i))

        print('Missing: {}/{}'.format(len(missing), counter))
        print(missing)

    elif args.task == 'combine':
        pairs = get_prots(args.docked_prot_file)
        random.shuffle(pairs)
        for protein, target, start in pairs[:5]:
            print(protein, target, start)
            if protein == 'Q86WV6':
                continue
            pair = '{}-to-{}'.format(target, start)
            protein_path = os.path.join(raw_root, protein)
            pair_path = os.path.join(protein_path, pair)
            grid_size = get_grid_size(pair_path, target, start)
            pose_path = os.path.join(pair_path, 'exhaustive_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size))
            correct_path = os.path.join(pose_path, 'correct_after_simple_filter')
            clash_path = os.path.join(correct_path, 'clash_data')
            file = os.path.join(correct_path, 'combined.csv')
            df = pd.read_csv(file)
            indices = [i for i in range(len(df))]
            grouped_indices = group_files(args.n, indices)
            dfs = []

            for i in range(len(grouped_indices)):
                pose_file = os.path.join(clash_path, 'pose_pred_data_{}.csv'.format(i))
                dfs.append(pd.read_csv(pose_file))

            df = pd.concat(dfs)
            df.to_csv(os.path.join(clash_path, 'combined.csv'), index=False)


if __name__ == "__main__":
    main()
