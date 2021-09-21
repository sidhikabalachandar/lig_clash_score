"""
The purpose of this code is to get the physics scores and the rmsds

It can be run on sherlock using
$ $SCHRODINGER/run python3 get_names.py /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d
"""

import argparse
import os
import pandas as pd
import time

import sys
sys.path.insert(1, '../sample/util')
from util import *
from prot_util import *
from schrod_replacement_util import *
from lig_util import *
from prepare_pockets import *

X_AXIS = [1.0, 0.0, 0.0]  # x-axis unit vector
Y_AXIS = [0.0, 1.0, 0.0]  # y-axis unit vector
Z_AXIS = [0.0, 0.0, 1.0]  # z-axis unit vector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=str, help='directory where raw data will be placed')
    parser.add_argument('--rmsd_cutoff', type=float, default=2.5, help='name of pose group subdir')
    parser.add_argument('--data_name', type=str, default='data', help='name of saved data folder')
    args = parser.parse_args()

    raw_root = os.path.join(args.root, 'raw')
    save_directory = os.path.join(args.root, 'ml_score')
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)
    save_directory = os.path.join(save_directory, args.data_name)
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    data = {'name': [], 'label': [], 'pocket_file': [], 'file': []}

    for protein, target, start in [('P02829', '2weq', '2yge'), ('P00797', '3own', '3d91'),
                                   ('C8B467', '5ult', '5uov'),
                                   ('P03368', '1gno', '1zp8'), ('P02829', '2fxs', '2weq'),
                                   ('P11838', '3wz6', '1gvx'),
                                   ('P00523', '4ybk', '2oiq'), ('P00519', '4twp', '5hu9'),
                                   ('P0DOX7', '6msy', '6mub')]:
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, pair)

        grid_size = get_grid_size(pair_path, target, start)
        group_name = 'test_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size)
        pose_path = os.path.join(pair_path, group_name)
        file = os.path.join(pose_path, 'poses_after_advanced_filter.csv')
        df = pd.read_csv(file)

        names = df['name'].to_list()

        if args.data_name == 'data':
            for name in names:
                pose_name = '{}_{}_{}'.format(protein, pair, name)
                rmsd = df[df['name'] == name]['rmsd'].iloc[0]
                if rmsd < args.rmsd_cutoff:
                    label = 0
                else:
                    label = 1
                data['name'].append(pose_name)
                data['label'].append(label)
                data['pocket_file'].append('{}_{}_pocket.mae'.format(protein, pair))
                data['file'].append('{}_{}_ligs.mae'.format(protein, pair))
        elif args.data_name == 'small_data':
            for name in names[:3]:
                pose_name = '{}_{}_{}'.format(protein, pair, name)
                rmsd = df[df['name'] == name]['rmsd'].iloc[0]
                if rmsd < args.rmsd_cutoff:
                    label = 0
                else:
                    label = 1
                data['name'].append(pose_name)
                data['label'].append(label)
                data['pocket_file'].append('{}_{}_pocket.mae'.format(protein, pair))
                data['file'].append('{}_{}_ligs.mae'.format(protein, pair))

    df = pd.DataFrame.from_dict(data)
    df.to_csv(os.path.join(save_directory, 'names.csv'), index=False)



if __name__=="__main__":
    main()