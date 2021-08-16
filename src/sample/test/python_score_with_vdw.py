"""
The purpose of this code is to create the cumulative frequency and bar graphs

It can be run on sherlock using
$ $SCHRODINGER/run python3 python_score_with_vdw.py group /home/users/sidhikab/lig_clash_score/src/sample/test/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/vdw_AMBER_parm99.defn --protein P00797 --target 3own --start 3d91 --index 0
"""

import argparse
import os
import pandas as pd
import schrodinger.structure as structure
import random
from schrodinger.structutils.transform import get_centroid
import math
import schrodinger.structutils.interactions.steric_clash as steric_clash
import sys
import numpy as np
sys.path.insert(1, '../util')
from util import *
from prot_util import *
from schrod_replacement_util import *
sys.path.insert(1, '../../../../physics_scoring')
from score_np import *
from read_vdw_params import *
sys.path.append('/home/users/sidhikab/docking')
from docking.docking_class import Docking_Set


X_AXIS = [1.0, 0.0, 0.0]  # x-axis unit vector
Y_AXIS = [0.0, 1.0, 0.0]  # y-axis unit vector
Z_AXIS = [0.0, 0.0, 1.0]  # z-axis unit vector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either align or search')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('vdw_param_file', type=str, help='directory where raw data will be placed')
    parser.add_argument('--protein', type=str, default='', help='protein name')
    parser.add_argument('--target', type=str, default='', help='target ligand name')
    parser.add_argument('--start', type=str, default='', help='start ligand name')
    parser.add_argument('--max_poses', type=int, default=100, help='maximum number of glide poses considered')
    parser.add_argument('--num_conformers', type=int, default=300, help='maximum number of conformers considered')
    parser.add_argument('--rotation_search_step_size', type=int, default=5, help='step size between each angle '
                                                                                 'checked, in degrees')
    parser.add_argument('--grid_size', type=int, default=1, help='grid size in positive and negative x, y, z '
                                                                 'directions')
    parser.add_argument('--target_clash_cutoff', type=int, default=20, help='clash cutoff between target protein and '
                                                                            'ligand pose')
    parser.add_argument('--intolerable_cutoff', type=int, default=0, help='cutoff of max num intolerable residues')
    parser.add_argument('--index', type=int, default=-1, help='index of pose file')
    parser.add_argument('--n', type=int, default=30, help='number of files processed in each job')
    parser.add_argument('--residue_cutoff', type=int, default=3, help='name of pose group subdir')
    args = parser.parse_args()

    random.seed(0)

    if args.task == 'all':
        counter = 0
        for protein, target, start in [('P00797', '3own', '3d91'), ('C8B467', '5ult', '5uov')]:
            pair = '{}-to-{}'.format(target, start)
            protein_path = os.path.join(args.raw_root, protein)
            pair_path = os.path.join(protein_path, pair)

            grid_size = get_grid_size(pair_path, target, start)
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
            names = df['name'].to_list()
            grouped_names = group_files(args.n, names)

            for i in range(len(grouped_names)):
                cmd = 'sbatch -p rondror -t 0:20:00 -o {} --wrap="$SCHRODINGER/run python3 python_score.py group {} {} ' \
                      '--protein {} --target {} --start {} --index {}"'
                counter += 1
                os.system(cmd.format(os.path.join(args.run_path, 'score_{}.out'.format(i)), args.run_path, args.raw_root,
                                     protein, target, start, i))

        print(counter)

    elif args.task == 'group':
        pair = '{}-to-{}'.format(args.target, args.start)
        protein_path = os.path.join(args.raw_root, args.protein)
        pair_path = os.path.join(protein_path, pair)

        grid_size = get_grid_size(pair_path, args.target, args.start)
        group_name = 'test_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size)
        pose_path = os.path.join(pair_path, group_name)

        conformer_file = os.path.join(pair_path, "aligned_to_start_with_hydrogen_conformers.mae")
        conformers = list(structure.StructureReader(conformer_file))

        grouped_path = os.path.join(pose_path, 'advanced_filtered_poses')
        dock_output_path = os.path.join(pose_path, 'dock_output')
        ground_truth_file = os.path.join(pair_path, 'ligand_poses', '{}_lig0.mae'.format(args.target))

        clash_path = os.path.join(pose_path, 'clash_data')
        dfs = []
        for file in os.listdir(clash_path):
            prefix = 'pose_pred_data'
            if file[:len(prefix)] == prefix:
                df = pd.read_csv(os.path.join(clash_path, file))
                filter_df = df[df['pred_num_intolerable'] < args.residue_cutoff]
                dfs.append(filter_df)

        df = pd.concat(dfs)

        protein_file = os.path.join(pair_path, '{}_prot.mae'.format(args.start))
        prot_s = list(structure.StructureReader(protein_file))[0]
        target_coord = prot_s.getXYZ(copy=True)
        target_charge = np.array([a.partial_charge for a in prot_s.atom])
        target_atom_type = [a.element for a in prot_s.atom]
        vdw_params = read_vdw_params(args.vdw_param_file)

        name = args.index
        docking_config = [{'folder': dock_output_path,
                           'name': name,
                           'grid_file': os.path.join(pair_path, '{}.zip'.format(pair)),
                           'prepped_ligand_file': os.path.join(grouped_path, '{}.mae'.format(name)),
                           'glide_settings': {'num_poses': 1, 'docking_method': 'inplace'},
                           'ligand_file': ground_truth_file}]
        dock_set = Docking_Set()
        results = dock_set.get_docking_gscores(docking_config, mode='multi')
        results_by_ligand = results[name]

        group_df = df[df['name'].isin(results_by_ligand)]

        for n in group_df['name'].to_list():
            conformer_index = df[df['name'] == n]['conformer_index'].iloc[0]
            c = conformers[conformer_index]
            old_coords = c.getXYZ(copy=True)
            grid_loc_x = df[df['name'] == n]['grid_loc_x'].iloc[0]
            grid_loc_y = df[df['name'] == n]['grid_loc_y'].iloc[0]
            grid_loc_z = df[df['name'] == n]['grid_loc_z'].iloc[0]
            translate_structure(c, grid_loc_x, grid_loc_y, grid_loc_z)
            conformer_center = list(get_centroid(c))
            coords = c.getXYZ(copy=True)
            rot_x = df[df['name'] == n]['rot_x'].iloc[0]
            rot_y = df[df['name'] == n]['rot_y'].iloc[0]
            rot_z = df[df['name'] == n]['rot_z'].iloc[0]

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
            c.title = n

            no_clash_atom_indices = []

            for i in c.getAtomIndices():
                clash = steric_clash.clash_volume(prot_s, struc2=c, atoms2=[i])
                if clash == 0:
                    no_clash_atom_indices.append(i)

            no_clash_c = c.extract(no_clash_atom_indices)

            ligand_coord = no_clash_c.getXYZ(copy=True)
            ligand_charge = np.array([a.partial_charge for a in no_clash_c.atom])
            ligand_atom_type = [a.element for a in no_clash_c.atom]
            python_score = physics_score(ligand_coord, ligand_charge, target_coord, target_charge,
                                         np.array(ligand_atom_type), np.array(target_atom_type), vdw_params=vdw_params)

            c.setXYZ(old_coords)

            print(python_score)
            return


if __name__=="__main__":
    main()