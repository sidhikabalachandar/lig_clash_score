"""
The purpose of this code is to create the cumulative frequency and bar graphs

It can be run on sherlock using
$ $SCHRODINGER/run python3 glide_poses.py all /home/users/sidhikab/lig_clash_score/src/sample/test/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/vdw_AMBER_parm99.defn
"""

import argparse
import os
import pandas as pd
import schrodinger.structure as structure
import random
import sys
import numpy as np
sys.path.insert(1, '../util')
from util import *
from prot_util import *
from schrod_replacement_util import *
sys.path.insert(1, '../../../../physics_scoring')
from score_np import *
from read_vdw_params import *


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
    args = parser.parse_args()

    random.seed(0)

    if args.task == 'all':
        for protein, target, start in [('P03368', '1gno', '1zp8'), ('P02829', '2fxs', '2weq'),
                                       ('P11838', '3wz6', '1gvx'), ('P00523', '4ybk', '2oiq'),
                                       ('P00519', '4twp', '5hu9'), ('P0DOX7', '6msy', '6mub')]:
            cmd = 'sbatch -p rondror -t 0:20:00 -o {} --wrap="$SCHRODINGER/run python3 glide_poses.py group {} {} {}' \
                  '--protein {} --target {} --start {}"'
            os.system(cmd.format(os.path.join(args.run_path, 'glide_{}_{}_{}.out'.format(protein, target, start)),
                                 args.run_path, args.raw_root, args.vdw_param_file, protein, target, start))

    elif args.task == 'group':
        pair = '{}-to-{}'.format(args.target, args.start)
        protein_path = os.path.join(args.raw_root, args.protein)
        pair_path = os.path.join(protein_path, pair)

        grid_size = get_grid_size(pair_path, args.target, args.start)
        group_name = 'test_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size)
        pose_path = os.path.join(pair_path, group_name)

        glide_df = pd.read_csv(os.path.join(pair_path, '{}.csv'.format(pair)))
        names = []
        score_no_vdws = []
        scores = []

        protein_file = os.path.join(pair_path, '{}_prot.mae'.format(args.start))
        prot_s = list(structure.StructureReader(protein_file))[0]
        target_coord = prot_s.getXYZ(copy=True)
        target_charge = np.array([a.partial_charge for a in prot_s.atom])
        target_atom_type = [a.element for a in prot_s.atom]
        vdw_params = read_vdw_params(args.vdw_param_file)

        for i in range(1, 100):
            name = '{}_lig{}'.format(args.target, i)
            pose_file = os.path.join(pair_path, 'ligand_poses', '{}.mae'.format(name))
            if not os.path.exists(pose_file):
                continue
            c = list(structure.StructureReader(pose_file))[0]
            ligand_coord = c.getXYZ(copy=True)
            ligand_charge = np.array([a.partial_charge for a in c.atom])
            ligand_atom_type = [a.element for a in c.atom]
            score_no_vdw = physics_score(ligand_coord, ligand_charge, target_coord, target_charge, ligand_atom_type,
                                  target_atom_type, vdw_scale=0)
            score = physics_score(ligand_coord, ligand_charge, target_coord, target_charge, np.array(ligand_atom_type),
                                  np.array(target_atom_type), vdw_params=vdw_params)
            names.append(name)
            score_no_vdws.append(score_no_vdw)
            scores.append(score)

        glide_df = glide_df.loc[glide_df['target'].isin(names)]
        glide_df['python_score'] = scores
        glide_df['python_score_no_vdw'] = score_no_vdws
        glide_df.to_csv(os.path.join(pose_path, 'glide_poses.csv'))

    elif args.task == 'remove':
        for protein, target, start in [('P02829', '2weq', '2yge'), ('P00797', '3own', '3d91'),
                                       ('C8B467', '5ult', '5uov')]:
            pair = '{}-to-{}'.format(target, start)
            protein_path = os.path.join(args.raw_root, protein)
            pair_path = os.path.join(protein_path, pair)

            grid_size = get_grid_size(pair_path, target, start)
            group_name = 'test_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size)
            pose_path = os.path.join(pair_path, group_name)
            combined_pose_file = os.path.join(pose_path, 'glide_poses.csv')
            os.system('rm -rf {}'.format(combined_pose_file))


if __name__=="__main__":
    main()