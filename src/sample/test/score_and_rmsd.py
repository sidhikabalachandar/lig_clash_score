"""
The purpose of this code is to get the physics scores and the rmsds

It can be run on sherlock using
$ $SCHRODINGER/run python3 score_and_rmsd.py run /home/users/sidhikab/lig_clash_score/src/sample/test/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw --protein P02829 --target 2weq --start 2yge  --max_num_concurrent_jobs 1
"""

import argparse
import os
from tqdm import tqdm

import sys
sys.path.append('/home/users/sidhikab/docking')
from docking.docking_class import Docking_Set
import schrodinger.structure as structure
import pandas as pd
from docking.utilities import score_no_vdW
import math
from schrodinger.structutils.transform import get_centroid
import sys
sys.path.insert(1, '../util')
from util import *
from prot_util import *
from schrod_replacement_util import *


X_AXIS = [1.0, 0.0, 0.0]  # x-axis unit vector
Y_AXIS = [0.0, 1.0, 0.0]  # y-axis unit vector
Z_AXIS = [0.0, 0.0, 1.0]  # z-axis unit vector


def run(args):
    """
    get scores and rmsds
    """

    pair = '{}-to-{}'.format(args.target, args.start)
    protein_path = os.path.join(args.raw_root, args.protein)
    pair_path = os.path.join(protein_path, pair)

    grid_size = get_grid_size(pair_path, args.target, args.start)
    group_name = 'test_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size)
    pose_path = os.path.join(pair_path, group_name)

    clash_path = os.path.join(pose_path, 'clash_data')
    dfs = []
    for file in os.listdir(clash_path):
        prefix = 'pose_pred_data'
        if file[:len(prefix)] == prefix:
            df = pd.read_csv(os.path.join(clash_path, file))
            filter_df = df[df['pred_num_intolerable'] < args.residue_cutoff]
            dfs.append(filter_df)

    df = pd.concat(dfs)
    indices = [i for i in df.index]
    grouped_indices = group_files(args.n, indices)

    conformer_file = os.path.join(pair_path, "aligned_to_start_with_hydrogen_conformers.mae")
    conformers = list(structure.StructureReader(conformer_file))

    grouped_path = os.path.join(pose_path, 'advanced_filtered_poses')
    dock_output_path = os.path.join(pose_path, 'dock_output')
    ground_truth_file = os.path.join(pair_path, 'ligand_poses', '{}_lig0.mae'.format(args.target))

    if not os.path.exists(grouped_path):
        os.mkdir(grouped_path)

    if not os.path.exists(dock_output_path):
        os.mkdir(dock_output_path)

    docking_config = []

    for j in range(len(grouped_indices)):
        file = os.path.join(grouped_path, '{}.mae'.format(j))
        with structure.StructureWriter(file) as filtered:
            for i in grouped_indices[j]:
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
                c.title = name
                filtered.append(c)
                c.setXYZ(old_coords)

        if not os.path.exists(os.path.join(dock_output_path, '{}.scor'.format(j))):
            docking_config.append({'folder': dock_output_path,
                                   'name': j,
                                   'grid_file': os.path.join(pair_path, '{}.zip'.format(pair)),
                                   'prepped_ligand_file': file,
                                   'glide_settings': {'num_poses': 1, 'docking_method': 'inplace'},
                                   'ligand_file': ground_truth_file})
        if len(docking_config) == args.max_num_concurrent_jobs:
            break

    print(len(docking_config))

    run_config = {'run_folder': args.run_path,
                  'group_size': 1,
                  'partition': 'rondror',
                  'dry_run': False}

    dock_set = Docking_Set()
    dock_set.run_docking_rmsd_delete(docking_config, run_config)


def check(raw_root, protein, target, start, group_name):
    """
    check if scores and rmsds were calculated
    :param docked_prot_file: (string) file listing proteins to process
    :param raw_root: (string) directory where raw data will be placed
    :return:
    """

    pair = '{}-to-{}'.format(args.target, args.start)
    protein_path = os.path.join(args.raw_root, args.protein)
    pair_path = os.path.join(protein_path, pair)

    grid_size = get_grid_size(pair_path, args.target, args.start)
    group_name = 'test_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size)
    pose_path = os.path.join(pair_path, group_name)

    grouped_path = os.path.join(pose_path, 'advanced_filtered_poses')
    dock_output_path = os.path.join(pose_path, 'dock_output')
    ground_truth_file = os.path.join(pair_path, 'ligand_poses', '{}_lig0.mae'.format(args.target))

    missing = []
    incomplete = []

    for file in os.listdir(grouped_path):
        suffix = '.mae'
        name = file[:-len(suffix)]
        docking_config = [{'folder': dock_output_path,
                           'name': name,
                           'grid_file': os.path.join(pair_path, '{}.zip'.format(pair)),
                           'prepped_ligand_file': file,
                           'glide_settings': {'num_poses': 1, 'docking_method': 'inplace'},
                           'ligand_file': ground_truth_file}]
        dock_set = Docking_Set()
        if not os.path.exists(os.path.join(dock_output_path, '{}.scor'.format(name))):
            if os.path.exists(os.path.join(dock_output_path, '{}.in'.format(name))):
                os.remove(os.path.join(dock_output_path, '{}.in'.format(name)))
            if os.path.exists(os.path.join(dock_output_path, '{}.log'.format(name))):
                os.remove(os.path.join(dock_output_path, '{}.log'.format(name)))
            if os.path.exists(os.path.join(dock_output_path, '{}_state.json'.format(name))):
                os.remove(os.path.join(dock_output_path, '{}_state.json'.format(name)))
            missing.append(name)
            continue
        else:
            if not os.path.exists(os.path.join(dock_output_path, '{}_rmsd.csv'.format(name))):
                print(os.path.join(dock_output_path, '{}_rmsd.csv'.format(name)))
                incomplete.append(name)
                continue
            results = dock_set.get_docking_gscores(docking_config, mode='multi')
            results_by_ligand = results[name]
            group = list(structure.StructureReader(file))
            if len(results_by_ligand.keys()) != len(group):
                print(len(results_by_ligand), len(group))
                incomplete.append(name)
                continue

    print('Missing', len(missing), '/', len(os.listdir(grouped_path)))
    print('Incomplete', len(incomplete), '/', len(os.listdir(grouped_path)) - len(missing))
    print(incomplete)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either run, check, or delete_json')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('--protein', type=str, default='', help='protein name')
    parser.add_argument('--target', type=str, default='', help='target ligand name')
    parser.add_argument('--start', type=str, default='', help='start ligand name')
    parser.add_argument('--n', type=int, default=4, help='number of protein, target, start groups processed in '
                                                         'group task')
    parser.add_argument('--max_num_concurrent_jobs', type=int, default=200, help='maximum number of concurrent jobs '
                                                                                 'that can be run on slurm at one time')
    parser.add_argument('--residue_cutoff', type=int, default=3, help='name of pose group subdir')
    args = parser.parse_args()

    if not os.path.exists(args.run_path):
        os.mkdir(args.run_path)

    if args.task == 'run':
        run(args)

    elif args.task == 'check':
        check(args.raw_root, args.protein, args.target, args.start, args.group_name)

    elif args.task == 'add_data':
        pair = '{}-to-{}'.format(args.target, args.start)
        protein_path = os.path.join(args.raw_root, args.protein)
        pair_path = os.path.join(protein_path, pair)
        pose_path = os.path.join(pair_path, args.group_name)
        grouped_pose_path = os.path.join(pose_path, 'grouped_poses')
        dock_output_path = os.path.join(pose_path, 'dock_output')
        ground_truth_file = os.path.join(pair_path, 'ligand_poses', '{}_lig0.mae'.format(args.target))

        df = pd.read_csv(os.path.join(pair_path, 'poses_after_pred_filter.csv'))
        in_place_data = {}

        for file in tqdm(os.listdir(dock_output_path), desc='docked files'):
            if file[-4:] == 'scor':
                name = file[:-5]
                docking_config = [{'folder': dock_output_path,
                                   'name': name,
                                   'grid_file': os.path.join(pair_path, '{}.zip'.format(pair)),
                                   'prepped_ligand_file': os.path.join(grouped_pose_path, '{}.maegz'.format(name)),
                                   'glide_settings': {'num_poses': 1, 'docking_method': 'inplace'},
                                   'ligand_file': ground_truth_file}]
                dock_set = Docking_Set()
                results = dock_set.get_docking_gscores(docking_config, mode='multi')
                results_by_ligand = results[name]
                for n in results_by_ligand:
                    glide_score = results_by_ligand[n][0]['Score']
                    score = score_no_vdW(results_by_ligand[n][0])
                    modified_score_no_vdw = score
                    if score > 20:
                        modified_score_no_vdw = 20
                    elif score < -20:
                        modified_score_no_vdw = -20

                    in_place_data[n] = (glide_score, score, modified_score_no_vdw)

        glide_scores = []
        score_no_vdws = []
        modified_score_no_vdws = []
        for name in df['name'].to_list():
            glide_scores.append(in_place_data[name][0])
            score_no_vdws.append(in_place_data[name][1])
            modified_score_no_vdws.append(in_place_data[name][2])

        df.to_csv(os.path.join(pair_path, 'poses_after_pred_filter.csv'))

if __name__=="__main__":
    main()