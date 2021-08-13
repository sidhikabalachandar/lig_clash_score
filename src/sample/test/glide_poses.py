"""
The purpose of this code is to create the cumulative frequency and bar graphs

It can be run on sherlock using
$ $SCHRODINGER/run python3 glide_poses.py /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
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


X_AXIS = [1.0, 0.0, 0.0]  # x-axis unit vector
Y_AXIS = [0.0, 1.0, 0.0]  # y-axis unit vector
Z_AXIS = [0.0, 0.0, 1.0]  # z-axis unit vector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    args = parser.parse_args()

    random.seed(0)

    for protein, target, start in [('P00797', '3own', '3d91'), ('C8B467', '5ult', '5uov')]:
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(args.raw_root, protein)
        pair_path = os.path.join(protein_path, pair)

        grid_size = get_grid_size(pair_path, target, start)
        group_name = 'test_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size)
        pose_path = os.path.join(pair_path, group_name)

        glide_df = pd.read_csv(os.path.join(pair_path, '{}.csv'.format(pair)))
        names = []
        scores = []

        protein_file = os.path.join(pair_path, '{}_prot.mae'.format(start))
        prot_s = list(structure.StructureReader(protein_file))[0]
        target_coord = prot_s.getXYZ(copy=True)
        target_charge = np.array([a.partial_charge for a in prot_s.atom])
        target_atom_type = [a.element for a in prot_s.atom]

        for i in range(1, 100):
            name = '{}_lig{}'.format(target, i)
            pose_file = os.path.join(pair_path, 'ligand_poses', '{}.mae'.format(name))
            c = list(structure.StructureReader(pose_file))[0]
            ligand_coord = c.getXYZ(copy=True)
            ligand_charge = np.array([a.partial_charge for a in c.atom])
            ligand_atom_type = [a.element for a in c.atom]
            score = physics_score(ligand_coord, ligand_charge, target_coord, target_charge, ligand_atom_type,
                                  target_atom_type, vdw_scale=0)
            names.append(name)
            scores.append(score)

        glide_df = glide_df.loc[glide_df['target'].isin(names)]
        glide_df['np_score_no_vdw'] = scores
        glide_df.to_csv(os.path.join(pose_path, 'glide_poses.csv'))


if __name__=="__main__":
    main()