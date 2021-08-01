"""
The purpose of this code is to first create the raw directory folder and include the following files
starting protein receptor
starting ligand
target ligand
glide pose viewer file

Then the top glide poses are added

Then the decoys are created

It can be run on sherlock using
$ $SCHRODINGER/run python3 test.py check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits/search_test_incorrect_glide_index.txt /home/users/sidhikab/lig_clash_score/src/sample/test/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d --protein P00797 --target 3own --start 3d91 --index 0
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
    parser.add_argument('--save_pred_path', type=str, help='prediction graph file')
    parser.add_argument('--save_true_path', type=str, help='true graph file')
    parser.add_argument('--target_clash_cutoff', type=int, default=0, help='clash cutoff between target protein and '
                                                                            'ligand pose')
    parser.add_argument('--intolerable_cutoff', type=int, default=0, help='cutoff of max num intolerable residues')
    parser.add_argument('--rmsd_cutoff', type=int, default=2.5,
                        help='rmsd accuracy cutoff between predicted ligand pose '
                             'and true ligand pose')
    parser.add_argument('--n', type=int, default=4, help='number of files processed in each job')
    parser.add_argument('--start_clash_cutoff', type=int, default=1, help='clash cutoff between start protein and '
                                                                          'ligand pose')
    args = parser.parse_args()
    random.seed(0)

    raw_root = os.path.join(args.root, 'raw')

    if not os.path.exists(args.run_path):
        os.mkdir(args.run_path)

    if args.task == 'all':
        pairs = get_prots(args.docked_prot_file)
        random.shuffle(pairs)
        counter = 0
        for protein, target, start in pairs[5:10]:
            if protein == 'Q86WV6':
                continue
            pair = '{}-to-{}'.format(target, start)
            protein_path = os.path.join(raw_root, protein)
            pair_path = os.path.join(protein_path, pair)
            grid_size = get_grid_size(pair_path, target, start)
            pose_path = os.path.join(pair_path, 'test_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size))
            prefix = 'exhaustive_search_poses_'
            files = [f for f in os.listdir(pose_path) if f[:len(prefix)] == prefix]
            grouped_files = group_files(args.n, files)

            for i in range(len(grouped_files)):
                cmd = 'sbatch -p rondror -t 1:00:00 -o {} --wrap="$SCHRODINGER/run python3 test.py group {} {} ' \
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
        pose_path = os.path.join(pair_path, 'test_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size))

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

        # pocket res only
        target_lig_file = os.path.join(pair_path, 'ligand_poses', '{}_lig0.mae'.format(args.target))
        target_lig = list(structure.StructureReader(target_lig_file))[0]
        get_pocket_res(prot_docking, target_lig, 6)
        res = get_res(prot_docking)


        # get files to loop over
        prefix = 'exhaustive_search_poses_'
        suffix = '.csv'
        files = [f for f in os.listdir(pose_path) if f[:len(prefix)] == prefix]
        grouped_files = group_files(args.n, files)

        for file in grouped_files[args.index]:
            # create feature dictionary
            clash_features = {'name': [], 'residue': [], 'bfactor': [], 'mcss': [], 'volume_docking': []}

            # get indices
            all_df = pd.read_csv(os.path.join(pose_path, file))
            df = all_df[all_df['start_clash'] < args.start_clash_cutoff]
            correct_df = df[df['rmsd'] <= args.rmsd_cutoff]

            incorrect_df = df[df['rmsd'] > args.rmsd_cutoff]
            incorrect_names = incorrect_df['name'].to_list()
            random.shuffle(incorrect_names)
            incorrect_names = incorrect_names[:300]
            subset_incorrect_df = incorrect_df.loc[incorrect_df['name'].isin(incorrect_names)]

            subset_df = pd.concat([correct_df, subset_incorrect_df])

            for i in subset_df.index:
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

                for r in res:
                    volume_docking = steric_clash.clash_volume(prot_docking, atoms1=r.getAtomIndices(), struc2=c)
                    if volume_docking != 0:
                        clash_features['name'].append(name)
                        clash_features['residue'].append(r.getAsl())
                        clash_features['bfactor'].append(normalizedBFactor(r, mean, stdev))
                        clash_features['mcss'].append(mcss)
                        clash_features['volume_docking'].append(volume_docking)

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

            for pos, idx in enumerate(subset_df.index):
                name = subset_df.loc[[idx]]['name'].iloc[0]
                name_df = out_clash_df[out_clash_df['name'] == name]
                if len(name_df) != 0:
                    pred = name_df['pred'].to_list()
                    subset_df.iat[pos, subset_df.columns.get_loc('pred_num_intolerable')] = sum(pred)
                    subset_df.iat[pos, subset_df.columns.get_loc('num_clash_docking')] = len(pred)

                # else
                # default set to 0

            pose_file = os.path.join(clash_path, 'pose_pred_data_{}.csv'.format(file_name))
            subset_df.to_csv(pose_file, index=False)

    elif args.task == 'check':
        pairs = get_prots(args.docked_prot_file)
        random.shuffle(pairs)
        missing = []
        counter = 0
        for protein, target, start in pairs[5:10]:
            if protein == 'Q86WV6':
                continue
            pair = '{}-to-{}'.format(target, start)
            protein_path = os.path.join(raw_root, protein)
            pair_path = os.path.join(protein_path, pair)
            grid_size = get_grid_size(pair_path, target, start)
            pose_path = os.path.join(pair_path, 'test_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size))
            clash_path = os.path.join(pose_path, 'clash_data')

            prefix = 'exhaustive_search_poses_'
            suffix = '.csv'
            files = [f for f in os.listdir(pose_path) if f[:len(prefix)] == prefix]

            for file in files:
                file_name = file[len(prefix):-len(suffix)]
                out_clash_file = os.path.join(clash_path, 'clash_data_{}.csv'.format(file_name))
                pose_file = os.path.join(clash_path, 'pose_pred_data_{}.csv'.format(file_name))
                counter += 1
                if not os.path.exists(out_clash_file):
                    missing.append((protein, target, start, file))
                    continue
                if not os.path.exists(pose_file):
                    missing.append((protein, target, start, file))

        print('Missing: {}/{}'.format(len(missing), counter))
        print(missing)

    elif args.task == 'remove':
        pairs = get_prots(args.docked_prot_file)
        random.shuffle(pairs)
        for protein, target, start in pairs[:5]:
            if protein == 'Q86WV6':
                continue
            pair = '{}-to-{}'.format(target, start)
            protein_path = os.path.join(raw_root, protein)
            pair_path = os.path.join(protein_path, pair)
            grid_size = get_grid_size(pair_path, target, start)
            pose_path = os.path.join(pair_path, 'test_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size))
            clash_path = os.path.join(pose_path, 'clash_data')
            os.system('rm -rf {}'.format(clash_path))


if __name__ == "__main__":
    main()
