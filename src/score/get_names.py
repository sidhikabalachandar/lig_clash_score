"""
The purpose of this code is to get the physics scores and the rmsds

It can be run on sherlock using
$ $SCHRODINGER/run python3 get_ligands.py all /home/users/sidhikab/lig_clash_score/src/score/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d --protein P02829 --target 2weq --start 2yge
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
    parser.add_argument('task', type=str, help='either run, check, or delete_json')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('root', type=str, help='directory where raw data will be placed')
    parser.add_argument('--protein', type=str, default='', help='protein name')
    parser.add_argument('--target', type=str, default='', help='target ligand name')
    parser.add_argument('--start', type=str, default='', help='start ligand name')
    parser.add_argument('--n', type=int, default=35, help='number of protein, target, start groups processed in '
                                                         'group task')
    parser.add_argument('--max_num_concurrent_jobs', type=int, default=200, help='maximum number of concurrent jobs '
                                                                                 'that can be run on slurm at one time')
    parser.add_argument('--residue_cutoff', type=int, default=3, help='name of pose group subdir')
    args = parser.parse_args()

    raw_root = os.path.join(args.root, 'raw')
    save_directory = os.path.join(args.root, 'ml_score')
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)
    save_directory = os.path.join(save_directory, 'data')
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    if not os.path.exists(args.run_path):
        os.mkdir(args.run_path)

    if args.task == 'all':
        for protein, target, start in [('P02829', '2weq', '2yge'), ('P00797', '3own', '3d91'),
                                       ('C8B467', '5ult', '5uov'),
                                       ('P03368', '1gno', '1zp8'), ('P02829', '2fxs', '2weq'),
                                       ('P11838', '3wz6', '1gvx'),
                                       ('P00523', '4ybk', '2oiq'), ('P00519', '4twp', '5hu9'),
                                       ('P0DOX7', '6msy', '6mub')]:
            cmd = 'sbatch -p rondror -t 1:00:00 -o {} --wrap="$SCHRODINGER/run python3 get_ligands.py group {} {} ' \
                  '--protein {} --target {} --start {}"'
            os.system(
                cmd.format(os.path.join(args.run_path, 'ligands_{}_{}_{}.out'.format(protein, target, start)),
                           args.run_path, args.root, protein, target, start))

    if args.task == 'group':
        pair = '{}-to-{}'.format(args.target, args.start)
        protein_path = os.path.join(raw_root, args.protein)
        pair_path = os.path.join(protein_path, pair)

        grid_size = get_grid_size(pair_path, args.target, args.start)
        group_name = 'test_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size)
        pose_path = os.path.join(pair_path, group_name)
        file = os.path.join(pose_path, 'poses_after_advanced_filter.csv')
        df = pd.read_csv(file)

        names = df['name'].to_list()

        data = {'name': [], 'label': [], 'pocket_file': [], 'file': []}

        for name in names:
            start_time = time.time()
            pose_name = '{}_{}_{}'.format(args.protein, pair, name)
            rmsd = df[df['name'] == name]['rmsd'].iloc[0]
            data['name'].append(pose_name)
            data['label'].append(rmsd)
            data['pocket_file'].append('{}_{}_pocket.mae'.format(args.protein, pair))
            data['file'].append('{}_{}_ligs.mae'.format(args.protein, pair))
            print(time.time() - start_time)
            return 

        df = pd.DataFrame.from_dict(data)
        df.to_csv(os.path.join(save_directory, 'names.csv'), index=False)



if __name__=="__main__":
    main()