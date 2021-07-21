"""
The purpose of this code is to first create the raw directory folder and include the following files
starting protein receptor
starting ligand
target ligand
glide pose viewer file

Then the top glide poses are added

Then the decoys are created

It can be run on sherlock using
$ $SCHRODINGER/run python3 test_subsample_incorrect.py all /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits/search_test_incorrect_glide_index.txt /home/users/sidhikab/lig_clash_score/src/sample/test/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d --protein P00797 --target 3own --start 3d91 --index 0
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
    parser.add_argument('--group_name', type=str, default='', help='name of pose group subdir')
    parser.add_argument('--n', type=int, default=10, help='number of poses to process in each job')
    parser.add_argument('--save_pred_path', type=str, help='prediction graph file')
    parser.add_argument('--save_true_path', type=str, help='true graph file')
    parser.add_argument('--target_clash_cutoff', type=int, default=0, help='clash cutoff between target protein and '
                                                                            'ligand pose')
    parser.add_argument('--intolerable_cutoff', type=int, default=0, help='cutoff of max num intolerable residues')
    parser.add_argument('--rmsd_cutoff', type=int, default=2.5,
                        help='rmsd accuracy cutoff between predicted ligand pose and true ligand pose')
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
            subsample_path = os.path.join(pose_path, 'subsample_incorrect')

            prefix = 'index_'
            files = [f for f in os.listdir(subsample_path) if f[:len(prefix)] == prefix]
            grouped_files = group_files(args.n, files)

            print(protein, target, start, len(files), len(grouped_files))
            for i in range(len(grouped_files)):
                cmd = 'sbatch -p rondror -t 0:30:00 -o {} --wrap="$SCHRODINGER/run python3 ' \
                      'test_subsample_incorrect.py group {} {} {} --protein {} --target {} --start {} --index {}"'
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
        subsample_path = os.path.join(pose_path, 'subsample_incorrect')
        clash_path = os.path.join(subsample_path, 'clash_data')
        if not os.path.exists(clash_path):
            os.mkdir(clash_path)

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

        prefix = 'index_'
        suffix = '.pkl'
        files = [f for f in os.listdir(subsample_path) if f[:len(prefix)] == prefix]
        grouped_files = group_files(args.n, files)

        for file in grouped_files[args.index]:
            file_name = file[len(prefix):-len(suffix)]
            pose_file = 'exhaustive_search_poses_{}.csv'.format(file_name)
            start_df = pd.read_csv(os.path.join(pose_path, pose_file))
            incorrect_df = start_df[start_df['rmsd'] > args.rmsd_cutoff]
            out_clash_file = os.path.join(clash_path, 'clash_data_{}.csv'.format(file_name))
            infile = open(os.path.join(subsample_path, 'index_{}.pkl'.format(file_name)), 'rb')
            indices = pickle.load(infile)

            for i in indices:
                start_time = time.time()
                name = incorrect_df.loc[[i]]['name'].iloc[0]
                conformer_index = incorrect_df.loc[[i]]['conformer_index'].iloc[0]
                c = conformers[conformer_index]
                old_coords = c.getXYZ(copy=True)
                grid_loc_x = incorrect_df.loc[[i]]['grid_loc_x'].iloc[0]
                grid_loc_y = incorrect_df.loc[[i]]['grid_loc_y'].iloc[0]
                grid_loc_z = incorrect_df.loc[[i]]['grid_loc_z'].iloc[0]
                translate_structure(c, grid_loc_x, grid_loc_y, grid_loc_z)
                conformer_center = list(get_centroid(c))
                coords = c.getXYZ(copy=True)
                rot_x = incorrect_df.loc[[i]]['rot_x'].iloc[0]
                rot_y = incorrect_df.loc[[i]]['rot_y'].iloc[0]
                rot_z = incorrect_df.loc[[i]]['rot_z'].iloc[0]
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
                first_loop = time.time() - start_time

            out_clash_df = pd.DataFrame.from_dict(clash_features)
            infile = open(os.path.join(args.root, 'clash_classifier.pkl'), 'rb')
            clf = pickle.load(infile)
            infile.close()
            if len(out_clash_df) != 0:
                pred = clf.predict(out_clash_df[['bfactor', 'mcss', 'volume_docking']])
            else:
                pred = []
            out_clash_df['pred'] = pred
            out_clash_df.to_csv(out_clash_file, index=False)

            # add res info to pose data frame
            zeros = [0 for _ in range(len(incorrect_df))]
            incorrect_df['pred_num_intolerable'] = zeros
            incorrect_df['num_clash_docking'] = zeros

            for i in indices:
                start_time = time.time()
                name = incorrect_df.loc[[i]]['name'].iloc[0]
                name_df = out_clash_df[out_clash_df['name'] == name]
                if len(name_df) != 0:
                    pred = name_df['pred'].to_list()
                    incorrect_df.iat[i, incorrect_df.columns.get_loc('pred_num_intolerable')] = sum(pred)
                    incorrect_df.iat[i, incorrect_df.columns.get_loc('num_clash_docking')] = len(pred)

                # else
                # default set to 0
                second_loop = time.time() - start_time

            incorrect_df = incorrect_df.iloc[indices, :]
            incorrect_df.to_csv(os.path.join(clash_path, 'pose_pred_data_{}.csv'.format(file_name)))
            # print(first_loop + second_loop)

    elif args.task == 'check':
        pairs = get_prots(args.docked_prot_file)
        random.shuffle(pairs)
        missing = []
        for protein, target, start in pairs[:5]:
            if protein == 'Q86WV6':
                continue
            pair = '{}-to-{}'.format(target, start)
            protein_path = os.path.join(raw_root, protein)
            pair_path = os.path.join(protein_path, pair)
            pose_path = os.path.join(pair_path, args.group_name)
            subsample_path = os.path.join(pose_path, 'subsample')
            clash_path = os.path.join(subsample_path, 'clash_data')
            prefix = 'index_'
            suffix = '.pkl'

            for file in os.listdir(subsample_path):
                if file[:len(prefix)] == prefix:
                    name = file[len(prefix):-len(suffix)]
                    if not os.path.exists(os.path.join(clash_path, 'clash_data_{}.csv'.format(name))) or \
                            not os.path.exists(os.path.join(clash_path, 'pose_pred_data_{}.csv'.format(name))):
                        missing.append((protein, target, start, name))

        print(len(missing))
        print(missing)

    elif args.task == 'combine':
        pairs = get_prots(args.docked_prot_file)
        random.shuffle(pairs)
        for protein, target, start in pairs[:5]:
            if protein == 'Q86WV6':
                continue
            pair = '{}-to-{}'.format(target, start)
            protein_path = os.path.join(raw_root, protein)
            pair_path = os.path.join(protein_path, pair)
            pose_path = os.path.join(pair_path, args.group_name)
            correct_path = os.path.join(pose_path, 'correct_after_simple_filter')
            clash_path = os.path.join(correct_path, 'clash_data')
            dfs = []
            for f in os.listdir(clash_path):
                file = os.path.join(clash_path, f)
                prefix = 'pose'
                if f[:len(prefix)] == prefix:
                    dfs.append(pd.read_csv(file))

            df = pd.concat(dfs)
            df.to_csv(os.path.join(clash_path, 'combined.csv'), index=False)


if __name__ == "__main__":
    main()
