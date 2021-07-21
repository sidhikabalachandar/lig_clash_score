"""
The purpose of this code is to first create the raw directory folder and include the following files
starting protein receptor
starting ligand
target ligand
glide pose viewer file

Then the top glide poses are added

Then the decoys are created

It can be run on sherlock using
$ $SCHRODINGER/run python3 test.py group /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits/search_test_incorrect_glide_index.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d --protein P00797 --target 3own --start 3d91 --index 0 --n 1
"""

import argparse
import os
import random
import pandas as pd
import math
import pickle
import schrodinger.structure as structure
from schrodinger.structutils.transform import get_centroid
import schrodinger.structutils.interactions.steric_clash as steric_clash
import time
import sys
sys.path.insert(1, '../util')
from util import *
from prot_util import *
from schrod_replacement_util import *


X_AXIS = [1.0, 0.0, 0.0]  # x-axis unit vector
Y_AXIS = [0.0, 1.0, 0.0]  # y-axis unit vector
Z_AXIS = [0.0, 0.0, 1.0]  # z-axis unit vector


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
                cmd = 'sbatch -p rondror -t 0:20:00 -o {} --wrap="$SCHRODINGER/run python3 test.py group {} {} ' \
                      '{} --protein {} --target {} --start {} --index {}"'
                counter += 1
                os.system(cmd.format(os.path.join(args.run_path, 'test_{}_{}_{}.out'.format(protein, pair, i)),
                                     args.docked_prot_file, args.run_path, args.root, protein, target, start, i))

        print(counter)

    elif args.task == "group":
        # important dirs
        pair = '{}-to-{}'.format(args.target, args.start)
        protein_path = os.path.join(raw_root, args.protein)
        pair_path = os.path.join(protein_path, pair)
        grid_size = get_grid_size(pair_path, args.target, args.start)
        pose_path = os.path.join(pair_path, 'exhaustive_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size))

        clash_path = os.path.join(pose_path, 'clash_data')
        if not os.path.exists(clash_path):
            os.mkdir(clash_path)

        # important structures
        prot_docking = list(structure.StructureReader(os.path.join(pair_path, '{}_prot.mae'.format(args.start))))[0]
        conformer_file = os.path.join(pair_path, "aligned_to_start_with_hydrogen_conformers.mae")
        conformers = list(structure.StructureReader(conformer_file))

        # bfactor preprocessing
        mean, stdev = bfactor_stats(prot_docking)

        # mcss preprocessing
        with open('{}/{}_mcss.csv'.format(pair_path, pair)) as f:
            mcss = int(f.readline().strip().split(',')[4])

        # clash preprocessing part 1 (prot_docking has all res)
        docking_dim, docking_origin = get_dim(prot_docking)

        # pocket res only
        target_lig_file = os.path.join(pair_path, 'ligand_poses', '{}_lig0.mae'.format(args.target))
        target_lig = list(structure.StructureReader(target_lig_file))[0]
        get_pocket_res(prot_docking, target_lig, 6)

        # clash preprocessing part 2 (prot_docking only has pocket res)
        prot_docking_grid, grids = get_grid(prot_docking, docking_dim, docking_origin)

        # get files to loop over
        prefix = 'exhaustive_search_poses_'
        suffix = '.csv'
        files = [f for f in os.listdir(pose_path) if f[:len(prefix)] == prefix]
        grouped_files = group_files(args.n, files)

        for file in grouped_files[args.index]:
            start_time = time.time()
            # create feature dictionary
            clash_features = {'name': [], 'residue': [], 'bfactor': [], 'mcss': [], 'volume_docking': []}

            # get indices
            df = pd.read_csv(os.path.join(pose_path, file))
            correct_df = df[df['rmsd'] <= args.rmsd_cutoff]
            correct_indices = [i for i in correct_df.index]

            incorrect_df = df[df['rmsd'] > args.rmsd_cutoff]
            incorrect_indices = [i for i in incorrect_df.index]
            random.shuffle(incorrect_indices)
            incorrect_indices = incorrect_indices[:300]
            incorrect_indices = sorted(incorrect_indices)
            all_indices = correct_indices + incorrect_indices
            subset_df = df.iloc[all_indices, :]

            for i in all_indices:
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

                displacement_vector = get_coords_array_from_list(conformer_center)
                to_origin_matrix = get_translation_matrix(-1 * displacement_vector)
                from_origin_matrix = get_translation_matrix(displacement_vector)
                rot_matrix_x = get_rotation_matrix(X_AXIS, math.radians(rot_x))
                rot_matrix_y = get_rotation_matrix(Y_AXIS, math.radians(rot_y))
                rot_matrix_z = get_rotation_matrix(Z_AXIS, math.radians(rot_z))
                new_coords = rotate_structure(coords, from_origin_matrix, to_origin_matrix, rot_matrix_x,
                                              rot_matrix_y, rot_matrix_z)

                # for clash features dictionary
                c.setXYZ(new_coords)

                for grid, r in grids:
                    volume_docking = steric_clash.clash_volume(prot_docking, atoms1=r.getAtomIndices(), struc2=c)
                    if volume_docking != 0:
                        correct_clash_features['name'].append(name)
                        correct_clash_features['residue'].append(r.getAsl())
                        correct_clash_features['bfactor'].append(normalizedBFactor(r, mean, stdev))
                        correct_clash_features['mcss'].append(mcss)
                        correct_clash_features['volume_docking'].append(volume_docking)

                c.setXYZ(old_coords)

            # run ML model
            out_clash_df = pd.DataFrame.from_dict(clash_features)
            infile = open(os.path.join(args.root, 'clash_classifier.pkl'), 'rb')
            clf = pickle.load(infile)
            infile.close()
            if len(out_clash_df) != 0:
                pred = clf.predict(out_clash_df[['bfactor', 'mcss', 'volume_docking']])
            else:
                pred = []
            out_clash_df['pred'] = pred
            file_name = file[len(prefix):-len(suffix)]
            out_clash_file = os.path.join(clash_path, 'clash_data_{}.csv'.format(file_name))
            out_clash_df.to_csv(out_clash_file, index=False)

            # add clash info to pose data frame
            zeros = [0 for _ in range(len(subset_df))]
            subset_df['pred_num_intolerable'] = zeros
            subset_df['num_clash_docking'] = zeros

            for i in all_indices:
                name = subset_df.loc[[i]]['name'].iloc[0]
                name_df = out_clash_df[out_clash_df['name'] == name]
                if len(name_df) != 0:
                    pred = name_df['pred'].to_list()
                    subset_df.iat[i, incorrect_df.columns.get_loc('pred_num_intolerable')] = sum(pred)
                    subset_df.iat[i, incorrect_df.columns.get_loc('num_clash_docking')] = len(pred)

                # else
                # default set to 0

            pose_file = os.path.join(clash_path, 'pose_pred_data_{}.csv'.format(file_name))
            subset_df.to_csv(pose_file, index=False)
            print(time.time() - start_time)

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
