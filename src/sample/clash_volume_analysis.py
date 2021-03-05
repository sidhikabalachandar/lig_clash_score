"""
The purpose of this code is to first create the raw directory folder and include the following files
starting protein receptor
starting ligand
target ligand
glide pose viewer file

Then the top glide poses are added

Then the decoys are created

It can be run on sherlock using
$ $SCHRODINGER/run python3 clash_volume_analysis.py group /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /home/users/sidhikab/lig_clash_score/reports/figures/clash_analysis.png --file /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw/O38732/2i0a-to-2q5k/grid_no_target_clash_cutoff/0_0_0_poses.csv --index 0
"""

import argparse
import os
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import random
import pandas as pd
import numpy as np
import math
import schrodinger.structure as structure
import schrodinger.structutils.interactions.steric_clash as steric_clash
import schrodinger.structutils.rmsd as rmsd
from schrodinger.structutils.transform import get_centroid

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either align or search')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('save_path', type=str, help='directory where raw data will be placed')
    parser.add_argument('--file', type=str, default='', help='grid point group index')
    parser.add_argument('--index', type=int, default=-1, help='grid point group index')
    parser.add_argument('--rmsd_cutoff', type=int, default=2, help='rmsd accuracy cutoff between predicted ligand pose '
                                                                   'and true ligand pose')
    args = parser.parse_args()

    protein = 'O38732'
    target = '2i0a'
    start = '2q5k'

    pair = '{}-to-{}'.format(target, start)
    protein_path = os.path.join(args.raw_root, protein)
    pair_path = os.path.join(protein_path, pair)
    pose_path = os.path.join(pair_path, 'grid_no_target_clash_cutoff')
    random.seed(0)

    if args.task == 'all':
        for file in os.listdir(pose_path):
            if file[-9:] == 'poses.csv':
                for i in range(10, 20):
                    cmd = 'sbatch -p owners -t 1:00:00 -o {} --wrap="$SCHRODINGER/run python3 clash_volume_analysis.py ' \
                          'group {} {} {} --file {} --index {}"'
                    os.system(cmd.format(os.path.join(args.run_path, '{}_{}'.format(file, i)), args.run_path, args.raw_root,
                                         args.save_path, os.path.join(pose_path, file), i))

    elif args.task == "group":
        prot = list(structure.StructureReader(os.path.join(pair_path, '{}_prot.mae'.format(target))))[0]
        target_lig_file = os.path.join(pair_path, 'ligand_poses', '{}_lig0.mae'.format(target))
        target_lig = list(structure.StructureReader(target_lig_file))[0]
        conformer_file = os.path.join(pair_path, "aligned_to_start_with_hydrogen_conformers.mae")
        conformers = list(structure.StructureReader(conformer_file))
        df = pd.read_csv(args.file)
        names = df['name'].to_list()
        random.shuffle(names)
        bad_clash_ls = []
        good_clash_ls = []
        for i in range(args.index * 50, args.index * 50 + 50):
            name = names[i]
            c = conformers[df[df['name'] == name]['conformer_index'].iloc[0]]
            translate_structure(c, df[df['name'] == name]['grid_loc_x'].iloc[0],
                                df[df['name'] == name]['grid_loc_y'].iloc[0],
                                df[df['name'] == name]['grid_loc_z'].iloc[0])
            conformer_center = list(get_centroid(c))
            coords = c.getXYZ(copy=True)
            new_coords = rotate_structure(coords, math.radians(df[df['name'] == name]['rot_x'].iloc[0]),
                                          math.radians(df[df['name'] == name]['rot_y'].iloc[0]),
                                          math.radians(df[df['name'] == name]['rot_z'].iloc[0]), conformer_center)
            c.setXYZ(new_coords)
            rmsd_val = rmsd.calculate_in_place_rmsd(c, c.getAtomIndices(), target_lig, target_lig.getAtomIndices())
            residues = []
            for clash in steric_clash.clash_iterator(prot, struc2=c):
                r = clash[0].getResidue()
                if r not in residues:
                    residues.append(r)
                    clash_vol = steric_clash.clash_volume(prot, atoms1=r.getAtomIndices(), struc2=c)
                    if clash_vol > 100:
                        print(r, clash_vol)
                        print(df[df['name'] == name])
                        with structure.StructureWriter('clash_pose.mae') as clash:
                            clash.append(c)
                        return
                    if rmsd_val < args.rmsd_cutoff:
                        good_clash_ls.append(clash_vol)
                    else:
                        bad_clash_ls.append(clash_vol)

        save_path = os.path.join(pose_path, 'clash_vols')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        name = args.file.split('/')[-1][:-4]
        outfile = open(os.path.join(save_path, '{}_index_{}.pkl'.format(name, args.index)), 'wb')
        pickle.dump([bad_clash_ls, good_clash_ls], outfile)

    elif args.task == "check":
        missing = []
        save_path = os.path.join(pair_path, 'grid_search_poses', 'clash_vols')
        for file in os.listdir(pose_path):
            index = file[:-6]
            if not os.path.exists(os.path.join(save_path, '{}.pkl'.format(index))):
                missing.append(index)

        print("Missing:", missing)

    elif args.task == "graph":
        bad_clash_ls = []
        good_clash_ls = []
        clash_path = os.path.join(pose_path, 'clash_vols')
        for file in os.listdir(clash_path):
            file = os.path.join(clash_path, file)
            infile = open(file, 'rb')
            data = pickle.load(infile)
            infile.close()
            bad_clash_ls.extend(data[0])
            good_clash_ls.extend(data[1])

        print(len(bad_clash_ls))
        print(len(good_clash_ls))

        bad_clash_ls = [x for x in bad_clash_ls if x < 100]
        good_clash_ls = [x for x in good_clash_ls if x < 100]

        fig, ax = plt.subplots()
        sns.distplot(bad_clash_ls, hist=True, label="intolerable clash")
        sns.distplot(good_clash_ls, hist=True, label="tolerable clash")
        plt.title('Clash Distributions for O38732 2i0a-to-2q5k')
        plt.xlabel('clash volume')
        plt.ylabel('frequency')
        ax.legend()
        fig.savefig(args.save_path)

    elif args.task == 'stats':
        num = 0
        for file in os.listdir(pose_path):
            if file[-9:] == 'poses.csv':
                df = pd.read_csv(os.path.join(pose_path, file))
                num += len(df)

        print(num)



if __name__ == "__main__":
    main()