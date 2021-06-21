"""
The purpose of this code is to first create the raw directory folder and include the following files
starting protein receptor
starting ligand
target ligand
glide pose viewer file

Then the top glide poses are added

Then the decoys are created

It can be run on sherlock using
$ $SCHRODINGER/run python3 ideal.py group /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d --protein O38732 --target 2i0a --start 2q5k --group_name exhaustive_grid_1_rotation_5 --save_path /home/users/sidhikab/lig_clash_score/reports/figures/pred_analysis.png --file_index 0 --name_index 0
"""

import argparse
import os
import random
import pandas as pd
import numpy as np
import math
from Bio import pairwise2
import statistics
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import schrodinger.structure as structure
from schrodinger.structutils.transform import get_centroid
import schrodinger.structutils.interactions.steric_clash as steric_clash
import schrodinger.structutils.build as build

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

'''
Get the amino acid sequence
:param file: .mae file for the structure
:return: the amino acid string for all amino acids in chain A
'''
def get_sequence_from_str(s):
    str = ''
    for m in list(s.molecule):
        for r in list(m.residue):
            str += r.getCode()
    return str

'''
This method finds the pairwise alignemnt between the amino acid strings of each pair of proteins
:param protein: name of the protein
:param seq_file: path to the file containing the amino acid sequence of the protein
:param save_folder: path to the location where the alignment string should be saved
:return:
'''
def compute_protein_alignments(seq_docking, seq_target):
    alignments = pairwise2.align.globalxx(seq_docking, seq_target)
    return alignments[0][0], alignments[0][1]

'''
This function gets the pdbcode, chain, resnum, and getCode() of every residue in the protein structure
It ignores any residues associated with the ligand
:param s: the protein structure 
:return: the list of every residue's pdbcode, chain, resnum, and getCode()
'''
def get_all_res_asl(s):
    r_list = []
    for m in list(s.molecule):
        for r in list(m.residue):
            r_list.append((r.getCode(), r.getAsl()))
    return r_list


'''
This function gets the pdbcode, chain, resnum, and getCode() of every residue in the protein structure
It ignores any residues associated with the ligand
:param s: the protein structure 
:return: the list of every residue's pdbcode, chain, resnum, and getCode()
'''
def get_all_res_atoms(s):
    r_list = []
    for m in list(s.molecule):
        for r in list(m.residue):
            r_list.append((r.getCode(), tuple(r.getAtomIndices())))
    return r_list

'''
Maps unique residue identifiers to list index in alignment string

:param alignment_string: (string) output from alignment program, contains one letter codes and dashes
	example: 'TE--S--T-'
:param r_list: list of unique identifiers of each residue in order of sequence
	number of residues in r_list must be equal to number of residues in alignment_string
:return: the map of residues to alignment_string index
'''
def map_residues_to_align_index(alignment_string, r_list):
    r_to_i_map = {}
    counter = 0
    for i in range(len(alignment_string)):
        if counter >= len(r_list):
            break
        if alignment_string[i] == r_list[counter][0]:
            r_to_i_map[r_list[counter]] = i
            counter += 1
    return r_to_i_map

'''
Maps unique residue identifiers to list index in alignment string

:param alignment_string: (string) output from alignment program, contains one letter codes and dashes
	example: 'TE--S--T-'
:param r_list: list of unique identifiers of each residue in order of sequence
	number of residues in r_list must be equal to number of residues in alignment_string
:return: the map of residues to alignment_string index
'''
def map_index_to_residue(alignment_string, r_list):
    i_to_r_map = {}
    counter = 0
    for i in range(len(alignment_string)):
        if counter >= len(r_list):
            break
        if alignment_string[i] == r_list[counter][0]:
            i_to_r_map[i] = r_list[counter]
            counter += 1
    return i_to_r_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either align or search')
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('root', type=str, help='directory where raw data will be placed')
    parser.add_argument('--file_index', type=int, default=-1, help='index of pose file')
    parser.add_argument('--name_index', type=int, default=-1, help='index of name group in pose file')
    parser.add_argument('--protein', type=str, default='', help='name of protein')
    parser.add_argument('--target', type=str, default='', help='name of target ligand')
    parser.add_argument('--start', type=str, default='', help='name of start ligand')
    parser.add_argument('--group_name', type=str, default='', help='name of pose group subdir')
    parser.add_argument('--n', type=int, default=2000, help='number of grid_points processed in each job')
    parser.add_argument('--save_path', type=str, help='directory where raw data will be placed')
    parser.add_argument('--target_clash_cutoff', type=int, default=20, help='clash cutoff between target protein and '
                                                                            'ligand pose')
    args = parser.parse_args()
    random.seed(0)

    raw_root = os.path.join(args.root, 'raw')

    if not os.path.exists(args.run_path):
        os.mkdir(args.run_path)

    if args.task == 'all':
        pair = '{}-to-{}'.format(args.target, args.start)
        protein_path = os.path.join(raw_root, args.protein)
        pair_path = os.path.join(protein_path, pair)
        pose_path = os.path.join(pair_path, args.group_name)
        prefix = 'exhaustive_search_poses'
        files = [f for f in os.listdir(pose_path) if f[:len(prefix)] == prefix]
        for i in range(len(files)):
            file = os.path.join(pose_path, 'exhaustive_search_poses_{}.csv'.format(i))
            df = pd.read_csv(file)
            grouped_files = group_files(args.n, df['name'].to_list())
            for j in range(len(grouped_files)):
                cmd = 'sbatch -p rondror -t 1:00:00 -o {} --wrap="$SCHRODINGER/run python3 ideal.py group {} {} {} ' \
                      '--protein {} --target {} --start {} --group_name {} --file_index {} --name_index {}"'
                os.system(cmd.format(os.path.join(args.run_path, 'ideal_{}_{}.out'.format(i, j)), args.docked_prot_file,
                                     args.run_path, args.root, args.protein, args.target, args.start,
                                     args.group_name, i, j))
                return

    elif args.task == "group":
        pair = '{}-to-{}'.format(args.target, args.start)
        protein_path = os.path.join(raw_root, args.protein)
        pair_path = os.path.join(protein_path, pair)
        pose_path = os.path.join(pair_path, args.group_name)
        clash_path = os.path.join(pose_path, 'ideal_clash_data')
        if not os.path.exists(clash_path):
            os.mkdir(clash_path)
        out_file = os.path.join(clash_path, 'clash_data_{}_{}.csv'.format(args.file_index, args.name_index))
        file = os.path.join(pose_path, 'exhaustive_search_poses_{}.csv'.format(args.file_index))
        df = pd.read_csv(file)
        grouped_files = group_files(args.n, df['name'].to_list())
        names = grouped_files[args.name_index]
        prot_target = list(structure.StructureReader(os.path.join(pair_path, '{}_prot.mae'.format(args.target))))[0]
        conformer_file = os.path.join(pair_path, "aligned_to_start_with_hydrogen_conformers.mae")
        conformers = list(structure.StructureReader(conformer_file))

        features = {'name': [], 'conformer_index': [], 'grid_loc_x': [], 'grid_loc_y': [], 'grid_loc_z': [],
                    'rot_x': [], 'rot_y': [], 'rot_z': [], 'residue': [], 'volume_target': []}
        for name in tqdm(names, desc='names'):
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
            residues = []
            for clash in steric_clash.clash_iterator(prot_target, struc2=c):
                r = clash[0].getResidue()
                if r not in residues:
                    residues.append(r)
                    volume_target = steric_clash.clash_volume(prot_target, atoms1=r.getAtomIndices(), struc2=c)
                    features['name'].append(name)
                    features['conformer_index'].append(conformer_index)
                    features['grid_loc_x'].append(grid_loc_x)
                    features['grid_loc_y'].append(grid_loc_y)
                    features['grid_loc_z'].append(grid_loc_z)
                    features['rot_x'].append(rot_x)
                    features['rot_y'].append(rot_y)
                    features['rot_z'].append(rot_z)
                    features['residue'].append(r.getAsl())
                    features['volume_target'].append(volume_target)
            c.setXYZ(old_coords)

        out_df = pd.DataFrame.from_dict(features)
        out_df.to_csv(out_file, index=False)

        group_df = df[df['name'].isin(names)]
        num_intolerable = []
        num_clash = []
        for name in tqdm(group_df['name'].to_list(), desc='names'):
            name_df = out_df[out_df['name'] == name]
            num_intolerable.append(len(name_df[name_df['volume_target'] > args.target_clash_cutoff]))
            num_clash.append(len(name_df))
        group_df['num_intolerable'] = num_intolerable
        group_df['num_clash'] = num_clash
        group_df.to_csv(os.path.join(clash_path, 'pose_data_{}_{}.csv'.format(args.file_index, args.name_index)))

    elif args.task == 'check':
        pair = '{}-to-{}'.format(args.target, args.start)
        protein_path = os.path.join(raw_root, args.protein)
        pair_path = os.path.join(protein_path, pair)
        pose_path = os.path.join(pair_path, args.group_name)
        clash_path = os.path.join(pose_path, 'clash_data')
        prefix = 'exhaustive_search_poses'
        files = [f for f in os.listdir(pose_path) if f[:len(prefix)] == prefix]
        clash_counter = 0
        pose_counter = 0
        for i in range(len(files)):
            file = os.path.join(pose_path, 'exhaustive_search_poses_{}.csv'.format(i))
            df = pd.read_csv(file)
            grouped_files = group_files(args.n, df['name'].to_list())
            for j in range(len(grouped_files)):
                out_file = os.path.join(clash_path, 'clash_data_{}_{}.csv'.format(i, j))
                if not os.path.exists(out_file):
                    clash_counter += 1

                out_file = os.path.join(clash_path, 'pose_pred_data_{}_{}.csv'.format(i, j))
                if not os.path.exists(out_file):
                    pose_counter += 1
        print('Missing clash:', clash_counter)
        print('Missing pose:', pose_counter)

    elif args.task == 'combine':
        clash_dfs = []
        pose_dfs = []
        pair = '{}-to-{}'.format(args.target, args.start)
        protein_path = os.path.join(raw_root, args.protein)
        pair_path = os.path.join(protein_path, pair)
        pose_path = os.path.join(pair_path, args.group_name)
        clash_path = os.path.join(pose_path, 'clash_data')
        for f in os.listdir(clash_path):
            file = os.path.join(clash_path, f)
            df = pd.read_csv(os.path.join(file))
            if f[:5] == 'clash':
                clash_dfs.append(df)
            else:
                pose_dfs.append(df)

        # df = pd.concat(clash_dfs)
        # df.to_csv(os.path.join(pose_path, 'combined_clash_data.csv'))

        df = pd.concat(pose_dfs)
        df.to_csv(os.path.join(pose_path, 'combined_pose_data.csv'))

    elif args.task == 'graph':
        pair = '{}-to-{}'.format(args.target, args.start)
        protein_path = os.path.join(raw_root, args.protein)
        pair_path = os.path.join(protein_path, pair)
        pose_path = os.path.join(pair_path, args.group_name)
        df = pd.read_csv(os.path.join(pose_path, 'combined_pose_data.csv'))
        df = df[df['num_pred_intolerable'] != 0]
        correct_df = df[df['rmsd'] <= 2]
        incorrect_df = df[df['rmsd'] > 2]
        print(len(correct_df), len(incorrect_df))
        names = incorrect_df['name'].to_list()
        random.shuffle(names)
        names = names[:5000]
        incorrect_df = incorrect_df[incorrect_df['name'].isin(names)]

        fig, ax = plt.subplots()
        sns.distplot(correct_df['num_pred_intolerable'].to_list(), hist=True, label="correct poses (rmsd <= 2 A)")
        sns.distplot(incorrect_df['num_pred_intolerable'].to_list(), hist=True, label="incorrect poses (rmsd > 2 A)")
        plt.title('Num pred intolerable for O38732 2i0a-to-2q5k')
        plt.xlabel('Num pred intolerable')
        plt.ylabel('frequency')
        ax.legend()
        fig.savefig(args.save_path)


if __name__=="__main__":
    main()