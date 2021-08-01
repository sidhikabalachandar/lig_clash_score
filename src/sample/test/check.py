"""
The purpose of this code is to create conformers

It can be run on sherlock using
$ $SCHRODINGER/run python3 check.py /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits/search_test_incorrect_glide_index.txt /home/users/sidhikab/lig_clash_score/src/sample/test/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
"""

import argparse
import os
import schrodinger.structure as structure
from schrodinger.structutils.transform import get_centroid
import schrodinger.structutils.interactions.steric_clash as steric_clash
import schrodinger.structutils.rmsd as rmsd
import random
import math
import pandas as pd
import sys
sys.path.insert(1, '../util')
from util import *
from prot_util import *
from schrod_replacement_util import *

X_AXIS = [1.0, 0.0, 0.0]  # x-axis unit vector
Y_AXIS = [0.0, 1.0, 0.0]  # y-axis unit vector
Z_AXIS = [0.0, 0.0, 1.0]  # z-axis unit vector
EPS = 1e-6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('--num_conformers', type=int, default=300, help='maximum number of conformers considered')
    parser.add_argument('--num_pairs', type=int, default=10, help='number of protein-ligand pairs considered')
    parser.add_argument('--grid_n', type=int, default=75, help='number of grid_points processed in each job')
    parser.add_argument('--grid_size', type=int, default=1, help='grid size in positive and negative x, y, z '
                                                                 'directions')
    parser.add_argument('--grid_file', type=str, default='', help='pickle file with grid data dictionary')
    parser.add_argument('--grid_search_step_size', type=int, default=2, help='step size between each grid point, in '
                                                                             'angstroms')
    parser.add_argument('--rotation_search_step_size', type=int, default=20, help='step size between each angle '
                                                                                  'checked, in degrees')
    parser.add_argument('--min_angle', type=int, default=0, help='min angle of rotation in degrees')
    parser.add_argument('--max_angle', type=int, default=360, help='min angle of rotation in degrees')
    parser.add_argument('--conformer_n', type=int, default=10, help='number of conformers processed in each job')
    parser.add_argument('--rmsd_cutoff', type=int, default=2.5,
                        help='rmsd accuracy cutoff between predicted ligand pose '
                             'and true ligand pose')
    parser.add_argument('--start_clash_cutoff', type=int, default=2, help='clash cutoff between start protein and '
                                                                          'ligand pose')
    args = parser.parse_args()

    random.seed(0)

    if not os.path.exists(args.run_path):
        os.mkdir(args.run_path)

    pairs = get_prots(args.docked_prot_file)
    random.shuffle(pairs)
    pair_index = random.choice([i for i in range(5, 10)])
    protein, target, start = pairs[pair_index]
    print(protein, target, start, pair_index)

    pair = '{}-to-{}'.format(target, start)
    protein_path = os.path.join(args.raw_root, protein)
    pair_path = os.path.join(protein_path, pair)
    grid_size = get_grid_size(pair_path, target, start)
    group_name = 'test_grid_{}_{}_rotation_{}_{}_{}_rmsd_{}'.format(grid_size, args.grid_search_step_size,
                                                                    args.min_angle, args.max_angle,
                                                                    args.rotation_search_step_size, args.rmsd_cutoff)
    pose_path = os.path.join(pair_path, group_name)

    # get grid index
    grouped_grid_locs = group_grid(args.grid_n, grid_size, 2)
    grid_index = random.choice([i for i in range(len(grouped_grid_locs))])

    # get conformer index
    conformer_file = os.path.join(pair_path, "aligned_to_start_with_hydrogen_conformers.mae")
    conformers = list(structure.StructureReader(conformer_file))[:args.num_conformers]
    conformer_indices = [i for i in range(len(conformers))]
    grouped_conformer_indices = group_files(args.conformer_n, conformer_indices)
    conformer_index = random.choice([i for i in range(len(grouped_conformer_indices))])

    pose_file = os.path.join(pose_path, 'exhaustive_search_poses_{}_{}.csv'.format(grid_index, conformer_index))
    print(pose_file)

    if not os.path.exists(pose_file):
        cmd = 'sbatch -p rondror -t 0:20:00 -o {} --wrap="$SCHRODINGER/run python3 search.py group {} {} {} ' \
              '--rotation_search_step_size {} --grid_size {} --grid_n {} --num_conformers {} ' \
              '--conformer_n {} --grid_index {} --conformer_index {} --protein {} --target {} --start {}"'
        out_file_name = 'search_{}_{}_{}_{}_{}.out'.format(protein, target, start, grid_index, conformer_index)
        os.system(
            cmd.format(os.path.join(args.run_path, out_file_name), args.docked_prot_file, args.run_path,
                       args.raw_root, args.rotation_search_step_size, args.grid_size, args.grid_n,
                       args.num_conformers, args.conformer_n, grid_index, conformer_index, protein, target, start))
    else:
        df = pd.read_csv(pose_file)
        tolerable_indices = [i for i in df.index]
        pose_index = random.choice(tolerable_indices)

        conformer_index = df.loc[[pose_index]]['conformer_index'].iloc[0]
        c = conformers[conformer_index]
        old_coords = c.getXYZ(copy=True)
        grid_loc_x = df.loc[[pose_index]]['grid_loc_x'].iloc[0]
        grid_loc_y = df.loc[[pose_index]]['grid_loc_y'].iloc[0]
        grid_loc_z = df.loc[[pose_index]]['grid_loc_z'].iloc[0]
        translate_structure(c, grid_loc_x, grid_loc_y, grid_loc_z)
        conformer_center = list(get_centroid(c))
        coords = c.getXYZ(copy=True)
        rot_x = df.loc[[pose_index]]['rot_x'].iloc[0]
        rot_y = df.loc[[pose_index]]['rot_y'].iloc[0]
        rot_z = df.loc[[pose_index]]['rot_z'].iloc[0]

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

        print('conformer_index: {}, grid_loc: ({}, {}, {}), rotation: ({}, {}, {})'.format(conformer_index, grid_loc_x,
                                                                                           grid_loc_y, grid_loc_z,
                                                                                           rot_x, rot_y, rot_z))

        with structure.StructureWriter('pose.mae') as save:
            save.append(c)

        df_start_clash = df.loc[[pose_index]]['start_clash'].iloc[0]
        df_target_clash = df.loc[[pose_index]]['target_clash'].iloc[0]
        df_rmsd = df.loc[[pose_index]]['rmsd'].iloc[0]

        # prots
        start_prot_file = os.path.join(pair_path, '{}_prot.mae'.format(start))
        start_prot = list(structure.StructureReader(start_prot_file))[0]

        target_prot_file = os.path.join(pair_path, '{}_prot.mae'.format(target))
        target_prot = list(structure.StructureReader(target_prot_file))[0]

        for m in list(start_prot.molecule):
            for r in list(m.residue):
                if r.resnum == 49:
                    res_s = r.extractStructure()

        for a in res_s.atom:
            if a.element == 'C':
                with structure.StructureWriter('prot_{}.mae'.format(a.index)) as save:
                    save.append(res_s.extract([a.index]))

        for a in c.atom:
            if a.element == 'H':
                with structure.StructureWriter('lig_{}.mae'.format(a.index)) as save:
                    save.append(c.extract([a.index]))

        # clash preprocessing
        start_prot_grid, start_origin = get_grid(start_prot)
        # target_prot_grid, target_origin = get_grid(target_prot)

        c_indices = [a.index for a in c.atom if a.element != 'H']

        # ground truth lig
        target_lig_file = os.path.join(pair_path, 'ligand_poses', '{}_lig0.mae'.format(target))
        target_lig = list(structure.StructureReader(target_lig_file))[0]

        # get non hydrogen atom indices for rmsd
        target_lig_indices = [a.index for a in target_lig.atom if a.element != 'H']

        rmsd_val = rmsd.calculate_in_place_rmsd(c, c_indices, target_lig, target_lig_indices)
        start_clash = get_clash(c, start_prot_grid, start_origin)
        # target_clash = get_clash(c, target_prot_grid, target_origin)

        volume_docking = steric_clash.clash_volume(start_prot, struc2=c)
        print('volume docking:', volume_docking)
        print('start_clash', start_clash)

        assert(df_start_clash == start_clash)
        # assert (df_target_clash == target_clash)
        assert (df_rmsd - rmsd_val < EPS)
        print('All correct!')

        c.setXYZ(old_coords)




if __name__ == "__main__":
    main()
