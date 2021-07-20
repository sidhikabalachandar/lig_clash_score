"""
The purpose of this code is to first create the raw directory folder and include the following files
starting protein receptor
starting ligand
target ligand
glide pose viewer file

Then the top glide poses are added

Then the decoys are created

It can be run on sherlock using
$ $SCHRODINGER/run python3 data.py check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/sample/train/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d --group_name train_grid_6_1_rotation_0_360_20 --index 0 --n 1
"""

import argparse
import os
import random
import pandas as pd
import math
import schrodinger.structure as structure
from schrodinger.structutils.transform import get_centroid
import schrodinger.structutils.interactions.steric_clash as steric_clash
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys
sys.path.insert(1, '../util')
from util import *
from prot_util import *
from schrod_replacement_util import *

X_AXIS = [1.0, 0.0, 0.0]  # x-axis unit vector
Y_AXIS = [0.0, 1.0, 0.0]  # y-axis unit vector
Z_AXIS = [0.0, 0.0, 1.0]  # z-axis unit vector


def run_group(protein, target, start, raw_root, args):
    # important dirs
    pair = '{}-to-{}'.format(target, start)
    protein_path = os.path.join(raw_root, protein)
    pair_path = os.path.join(protein_path, pair)
    pose_path = os.path.join(pair_path, args.group_name)
    df = pd.read_csv(os.path.join(pose_path, 'poses.csv'))

    # save path
    out_clash_file = os.path.join(pose_path, 'res_data.csv')

    # prots
    prot_docking = list(structure.StructureReader(os.path.join(pair_path, '{}_prot.mae'.format(start))))[0]
    prot_target = list(structure.StructureReader(os.path.join(pair_path, '{}_prot.mae'.format(target))))[0]

    # conformers
    conformer_file = os.path.join(pair_path, "aligned_to_start_with_hydrogen_conformers.mae")
    conformers = list(structure.StructureReader(conformer_file))

    # bfactor info
    mean, stdev = bfactor_stats(prot_docking)

    # mcss info
    with open('{}/{}_mcss.csv'.format(pair_path, pair)) as f:
        mcss = int(f.readline().strip().split(',')[4])

    # map residues between docking and target proteins
    seq_docking = get_sequence_from_str(prot_docking)
    seq_target = get_sequence_from_str(prot_target)
    alignment_docking, alignment_target = compute_protein_alignments(seq_docking, seq_target)
    r_list_docking = get_all_res_asl(prot_docking)
    r_list_target = get_all_res_atoms(prot_target)
    r_to_i_map_docking = map_residues_to_align_index(alignment_docking, r_list_docking)
    i_to_r_map_target = map_index_to_residue(alignment_target, r_list_target)

    # create feature dictionary
    features = {'name': [], 'residue': [], 'bfactor': [], 'mcss': [], 'volume_docking': [], 'volume_target': []}

    for i in df.index:
        name = df.loc[[i]]['name'].iloc[0]

        # get conformer
        conformer_index = df.loc[[i]]['conformer_index'].iloc[0]
        c = conformers[conformer_index]

        # translate
        old_coords = c.getXYZ(copy=True)
        grid_loc_x = df.loc[[i]]['grid_loc_x'].iloc[0]
        grid_loc_y = df.loc[[i]]['grid_loc_y'].iloc[0]
        grid_loc_z = df.loc[[i]]['grid_loc_z'].iloc[0]
        translate_structure(c, grid_loc_x, grid_loc_y, grid_loc_z)

        # rotate
        conformer_center = list(get_centroid(c))
        coords = c.getXYZ(copy=True)
        rot_x = df.loc[[i]]['rot_x'].iloc[0]
        rot_y = df.loc[[i]]['rot_y'].iloc[0]
        rot_z = df.loc[[i]]['rot_z'].iloc[0]

        displacement_vector = get_coords_array_from_list(conformer_center)
        to_origin_matrix = get_translation_matrix(-1 * displacement_vector)
        from_origin_matrix = get_translation_matrix(displacement_vector)
        rot_matrix_x = get_rotation_matrix(X_AXIS, math.radians(rot_x))
        rot_matrix_y = get_rotation_matrix(Y_AXIS, math.radians(rot_y))
        rot_matrix_z = get_rotation_matrix(Z_AXIS, math.radians(rot_z))
        new_coords = rotate_structure(coords, from_origin_matrix, to_origin_matrix, rot_matrix_x,
                                      rot_matrix_y, rot_matrix_z)
        c.setXYZ(new_coords)

        # find all residues in docking protein with clash
        residues = []
        for clash in steric_clash.clash_iterator(prot_docking, struc2=c):

            # only include residue if we haven't already included it
            r = clash[0].getResidue()
            if r not in residues:
                residues.append(r)

                # only include residue if there is a non-negligible clash in docking protein
                volume_docking = steric_clash.clash_volume(prot_docking, atoms1=r.getAtomIndices(), struc2=c)
                if volume_docking > 5:

                    # find corresponding residue in target protein
                    id = (r.getCode(), r.getAsl())
                    i = r_to_i_map_docking[id]
                    if i in i_to_r_map_target:
                        atoms = list(i_to_r_map_target[i][1])
                        volume_target = steric_clash.clash_volume(prot_target, atoms1=atoms, struc2=c)
                        features['name'].append(name)
                        features['residue'].append(r.getAsl())
                        features['bfactor'].append(normalizedBFactor(r, mean, stdev))
                        features['mcss'].append(mcss)
                        features['volume_docking'].append(volume_docking)
                        features['volume_target'].append(volume_target)

        c.setXYZ(old_coords)

    out_df = pd.DataFrame.from_dict(features)
    out_df.to_csv(out_clash_file, index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either align or search')
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('root', type=str, help='directory where raw data will be placed')
    parser.add_argument('--index', type=int, default=-1, help='grid point group index')
    parser.add_argument('--n', type=int, default=6, help='number of grid_points processed in each job')
    parser.add_argument('--group_name', type=str, default='', help='name of pose group subdir')
    parser.add_argument('--rmsd_cutoff', type=float, default=2.5, help='name of pose group subdir')
    args = parser.parse_args()
    random.seed(0)

    raw_root = os.path.join(args.root, 'raw')

    if not os.path.exists(args.run_path):
        os.mkdir(args.run_path)

    if args.task == 'all':
        # pairs = get_prots(args.docked_prot_file)
        pairs = [('P61823', '1rnm', '1afl'), ('P61964', '4ql1', '6dar'), ('P61964', '5m23', '5sxm'), ('P61964', '5m25', '5sxm'), ('P61964', '5sxm', '6dar'), ('P61964', '6d9x', '5m25'), ('P61964', '6dai', '6dak'), ('P61964', '6dak', '6dai'), ('P61964', '6dar', '6dai'), ('P62508', '2p7a', '2p7g'), ('P62508', '2p7g', '2p7z'), ('P62617', '2amt', '3elc'), ('P62617', '2gzl', '3elc'), ('P62617', '3elc', '2amt'), ('P62937', '5t9u', '6gjj'), ('P62937', '5t9w', '5ta4'), ('P62937', '5t9z', '6gjm'), ('P62937', '6gji', '5t9w'), ('P62937', '6gjj', '5ta4'), ('P62937', '6gjl', '6gjm'), ('P62937', '6gjm', '5t9z'), ('P62937', '6gjn', '5t9z'), ('P62942', '1d7i', '1fkf'), ('P62942', '1d7j', '1fkb'), ('P62993', '3s8l', '3s8o'), ('P62993', '3s8n', '3ov1'), ('P62993', '3s8o', '3ove'), ('P63086', '4qyy', '6cpw'), ('P63086', '6cpw', '4qyy'), ('P64012', '3zhx', '3zi0'), ('P64012', '3zi0', '3zhx'), ('P66034', '2c92', '2c97'), ('P66034', '2c94', '2c97'), ('P66034', '2c97', '2c94'), ('P66992', '3qqs', '3r88'), ('P66992', '3r88', '4m0r'), ('P66992', '4owm', '4m0r'), ('P66992', '4owv', '3twp'), ('P68400', '2zjw', '5h8e'), ('P68400', '3bqc', '5cu4'), ('P68400', '3h30', '3pe2'), ('P68400', '3pe1', '5cu4'), ('P68400', '3pe2', '3pe1'), ('P68400', '5cqu', '5cu4'), ('P68400', '5csp', '3h30'), ('P68400', '5cu4', '3pe1'), ('P68400', '5h8e', '5csp'), ('P69834', '5mrm', '5mro'), ('P69834', '5mro', '5mrp'), ('P69834', '5mrp', '5mro'), ('P71094', '2jke', '2zq0'), ('P71094', '2jkp', '2zq0'), ('P71094', '2zq0', '2jkp'), ('P71447', '1z4o', '2wf5'), ('P71447', '2wf5', '1z4o'), ('P76141', '4l4z', '4l51'), ('P76141', '4l50', '4l51'), ('P76141', '4l51', '4l50'), ('P76637', '1ec9', '1ecq'), ('P76637', '1ecq', '1ec9'), ('P78536', '2oi0', '3l0v'), ('P78536', '3b92', '3le9'), ('P78536', '3ewj', '3le9'), ('P78536', '3kmc', '3le9'), ('P78536', '3l0v', '3lea'), ('P78536', '3le9', '3ewj'), ('P78536', '3lea', '3le9'), ('P80188', '3dsz', '3tf6'), ('P80188', '3tf6', '3dsz'), ('P84887', '2hjb', '2q7q'), ('P84887', '2q7q', '2hjb'), ('P95607', '3i4y', '3i51'), ('P95607', '3i51', '3i4y'), ('P96257', '6h1u', '6h2t'), ('P96257', '6h2t', '6h1u'), ('P98170', '2vsl', '4j46'), ('P98170', '3cm2', '3hl5'), ('P98170', '3hl5', '4j48'), ('P98170', '4j44', '4j45'), ('P98170', '4j45', '4j44'), ('P98170', '4j46', '3hl5'), ('P98170', '4j47', '4j44'), ('P98170', '4j48', '3cm2'), ('P9WMC0', '5f08', '5j3l'), ('P9WMC0', '5f0f', '5j3l'), ('P9WMC0', '5j3l', '5f08'), ('P9WPQ5', '6cvf', '6czc'), ('P9WPQ5', '6czb', '6cze'), ('P9WPQ5', '6czc', '6czb'), ('P9WPQ5', '6cze', '6czc'), ('Q00972', '3tz0', '4h7q'), ('Q00972', '4dzy', '4h85'), ('Q00972', '4h7q', '4h85'), ('Q00972', '4h81', '4h7q'), ('Q00972', '4h85', '4dzy'), ('Q00987', '4erf', '4wt2')]
        grouped_files = group_files(args.n, pairs)
        counter = 0
        for i in range(len(grouped_files)):
            cmd = 'sbatch -p rondror -t 0:30:00 -o {} --wrap="$SCHRODINGER/run python3 data.py group {} {} {} ' \
                  '--index {} --n {} --group_name {}"'
            os.system(cmd.format(os.path.join(args.run_path, 'data_{}.out'.format(i)), args.docked_prot_file,
                                 args.run_path, args.root, i, args.n, args.group_name))
            counter += 1

        print(counter)

    elif args.task == "group":
        # pairs = get_prots(args.docked_prot_file)
        pairs = [('P61823', '1rnm', '1afl'), ('P61964', '4ql1', '6dar'), ('P61964', '5m23', '5sxm'),
                 ('P61964', '5m25', '5sxm'), ('P61964', '5sxm', '6dar'), ('P61964', '6d9x', '5m25'),
                 ('P61964', '6dai', '6dak'), ('P61964', '6dak', '6dai'), ('P61964', '6dar', '6dai'),
                 ('P62508', '2p7a', '2p7g'), ('P62508', '2p7g', '2p7z'), ('P62617', '2amt', '3elc'),
                 ('P62617', '2gzl', '3elc'), ('P62617', '3elc', '2amt'), ('P62937', '5t9u', '6gjj'),
                 ('P62937', '5t9w', '5ta4'), ('P62937', '5t9z', '6gjm'), ('P62937', '6gji', '5t9w'),
                 ('P62937', '6gjj', '5ta4'), ('P62937', '6gjl', '6gjm'), ('P62937', '6gjm', '5t9z'),
                 ('P62937', '6gjn', '5t9z'), ('P62942', '1d7i', '1fkf'), ('P62942', '1d7j', '1fkb'),
                 ('P62993', '3s8l', '3s8o'), ('P62993', '3s8n', '3ov1'), ('P62993', '3s8o', '3ove'),
                 ('P63086', '4qyy', '6cpw'), ('P63086', '6cpw', '4qyy'), ('P64012', '3zhx', '3zi0'),
                 ('P64012', '3zi0', '3zhx'), ('P66034', '2c92', '2c97'), ('P66034', '2c94', '2c97'),
                 ('P66034', '2c97', '2c94'), ('P66992', '3qqs', '3r88'), ('P66992', '3r88', '4m0r'),
                 ('P66992', '4owm', '4m0r'), ('P66992', '4owv', '3twp'), ('P68400', '2zjw', '5h8e'),
                 ('P68400', '3bqc', '5cu4'), ('P68400', '3h30', '3pe2'), ('P68400', '3pe1', '5cu4'),
                 ('P68400', '3pe2', '3pe1'), ('P68400', '5cqu', '5cu4'), ('P68400', '5csp', '3h30'),
                 ('P68400', '5cu4', '3pe1'), ('P68400', '5h8e', '5csp'), ('P69834', '5mrm', '5mro'),
                 ('P69834', '5mro', '5mrp'), ('P69834', '5mrp', '5mro'), ('P71094', '2jke', '2zq0'),
                 ('P71094', '2jkp', '2zq0'), ('P71094', '2zq0', '2jkp'), ('P71447', '1z4o', '2wf5'),
                 ('P71447', '2wf5', '1z4o'), ('P76141', '4l4z', '4l51'), ('P76141', '4l50', '4l51'),
                 ('P76141', '4l51', '4l50'), ('P76637', '1ec9', '1ecq'), ('P76637', '1ecq', '1ec9'),
                 ('P78536', '2oi0', '3l0v'), ('P78536', '3b92', '3le9'), ('P78536', '3ewj', '3le9'),
                 ('P78536', '3kmc', '3le9'), ('P78536', '3l0v', '3lea'), ('P78536', '3le9', '3ewj'),
                 ('P78536', '3lea', '3le9'), ('P80188', '3dsz', '3tf6'), ('P80188', '3tf6', '3dsz'),
                 ('P84887', '2hjb', '2q7q'), ('P84887', '2q7q', '2hjb'), ('P95607', '3i4y', '3i51'),
                 ('P95607', '3i51', '3i4y'), ('P96257', '6h1u', '6h2t'), ('P96257', '6h2t', '6h1u'),
                 ('P98170', '2vsl', '4j46'), ('P98170', '3cm2', '3hl5'), ('P98170', '3hl5', '4j48'),
                 ('P98170', '4j44', '4j45'), ('P98170', '4j45', '4j44'), ('P98170', '4j46', '3hl5'),
                 ('P98170', '4j47', '4j44'), ('P98170', '4j48', '3cm2'), ('P9WMC0', '5f08', '5j3l'),
                 ('P9WMC0', '5f0f', '5j3l'), ('P9WMC0', '5j3l', '5f08'), ('P9WPQ5', '6cvf', '6czc'),
                 ('P9WPQ5', '6czb', '6cze'), ('P9WPQ5', '6czc', '6czb'), ('P9WPQ5', '6cze', '6czc'),
                 ('Q00972', '3tz0', '4h7q'), ('Q00972', '4dzy', '4h85'), ('Q00972', '4h7q', '4h85'),
                 ('Q00972', '4h81', '4h7q'), ('Q00972', '4h85', '4dzy'), ('Q00987', '4erf', '4wt2')]
        grouped_files = group_files(args.n, pairs)

        for protein, target, start in grouped_files[args.index]:
            start_time = time.time()
            run_group(protein, target, start, raw_root, args)
            print(time.time() - start_time)

    elif args.task == 'check':
        pairs = get_prots(args.docked_prot_file)
        unfinished = []

        for protein, target, start in pairs:
            pair = '{}-to-{}'.format(target, start)
            protein_path = os.path.join(raw_root, protein)
            pair_path = os.path.join(protein_path, pair)
            pose_path = os.path.join(pair_path, args.group_name)
            out_clash_file = os.path.join(pose_path, 'res_data.csv')
            if not os.path.exists(out_clash_file):
                unfinished.append((protein, target, start))

        print("Missing:", len(unfinished), "/", len(pairs))
        print(unfinished)

    elif args.task == 'all_combine':
        pairs = get_prots(args.docked_prot_file)
        grouped_files = group_files(args.n, pairs)
        counter = 0
        for i in range(len(grouped_files)):
            cmd = 'sbatch -p rondror -t 0:30:00 -o {} --wrap="$SCHRODINGER/run python3 data.py group_combine {} {} {} ' \
                  '--index {} --n {} --group_name {}"'
            os.system(cmd.format(os.path.join(args.run_path, 'data_{}.out'.format(i)), args.docked_prot_file,
                                 args.run_path, args.root, i, args.n, args.group_name))
            counter += 1

        print(counter)

    elif args.task == 'group_combine':
        pairs = get_prots(args.docked_prot_file)
        grouped_files = group_files(args.n, pairs)
        dfs = []

        for protein, target, start in grouped_files[args.index]:
            pair = '{}-to-{}'.format(target, start)
            protein_path = os.path.join(raw_root, protein)
            pair_path = os.path.join(protein_path, pair)
            pose_path = os.path.join(pair_path, args.group_name)
            out_clash_file = os.path.join(pose_path, 'res_data.csv')
            df = pd.read_csv(out_clash_file)
            df['protein'] = [protein for _ in range(len(df))]
            df['target'] = [target for _ in range(len(df))]
            df['start'] = [start for _ in range(len(df))]
            dfs.append(df)

        combined_df = pd.concat(dfs)
        save_path = os.path.join(args.root, 'res_data')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        combined_df.to_csv(os.path.join(save_path, 'combined_res_data_{}.csv'.format(args.index)))

    elif args.task == 'combine':
        save_path = os.path.join(args.root, 'res_data')
        dfs = []

        for file in os.listdir(save_path):
            print(file)
            df = pd.read_csv(os.path.join(save_path, file))
            dfs.append(df)

        combined_df = pd.concat(dfs)
        combined_df.to_csv(os.path.join(args.root, 'combined_res_data.csv'.format(args.index)))

    elif args.task == 'delete':
        pairs = get_prots(args.docked_prot_file)

        for protein, target, start in pairs:
            pair = '{}-to-{}'.format(target, start)
            protein_path = os.path.join(raw_root, protein)
            pair_path = os.path.join(protein_path, pair)
            pose_path = os.path.join(pair_path, args.group_name)
            out_clash_file = os.path.join(pose_path, 'res_data.csv')
            if os.path.exists(out_clash_file):
                os.remove(out_clash_file)

        # os.remove(os.path.join(args.root, 'combined_clash_data.csv'))

    elif args.task == 'clash':
        pairs = get_prots(args.docked_prot_file)

        clashes = []
        for protein, target, start in pairs:
            pair = '{}-to-{}'.format(target, start)
            protein_path = os.path.join(raw_root, protein)
            pair_path = os.path.join(protein_path, pair)
            pose_file = os.path.join(pair_path, 'train_grid_6_1_rotation_0_360_20.csv')
            df = pd.read_csv(pose_file)
            clashes.extend(df['schrod_target_clash'].tolist())

        fig, ax = plt.subplots()
        sns.distplot(clashes, hist=True)
        plt.title('Clash Distribution')
        plt.xlabel('clash volume')
        plt.ylabel('frequency')
        ax.legend()
        fig.savefig('custom_clash.png')


if __name__=="__main__":
    main()