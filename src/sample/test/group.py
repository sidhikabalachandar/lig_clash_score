"""
The purpose of this code is to create conformers

It can be run on sherlock using
$ $SCHRODINGER/run python3 group.py all_combine /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits/search_test_incorrect_glide_index.txt /home/users/sidhikab/lig_clash_score/src/sample/test/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw --protein P00797 --target 3own --start 3d91 --index 0 --n 1
"""

import argparse
import random
import os
import pandas as pd
import time
import sys
sys.path.insert(1, '../util')
from util import *
from prot_util import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either align or search')
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('raw_root', type=str, help='directory where script and output files will be written')
    parser.add_argument('--rmsd_cutoff', type=int, default=2.5,
                        help='rmsd accuracy cutoff between predicted ligand pose '
                             'and true ligand pose')
    parser.add_argument('--protein', type=str, default='', help='name of protein')
    parser.add_argument('--target', type=str, default='', help='name of target ligand')
    parser.add_argument('--start', type=str, default='', help='name of start ligand')
    parser.add_argument('--n', type=int, default=30, help='num jobs per ligand')
    parser.add_argument('--index', type=int, default=-1, help='index of job')
    args = parser.parse_args()
    random.seed(0)

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
            protein_path = os.path.join(args.raw_root, protein)
            pair_path = os.path.join(protein_path, pair)
            grid_size = get_grid_size(pair_path, target, start)
            pose_path = os.path.join(pair_path, 'exhaustive_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size))
            prefix = 'exhaustive_search_poses_'
            files = [f for f in os.listdir(pose_path) if f[:len(prefix)] == prefix]
            grouped_files = group_files(args.n, files)

            for i in range(len(grouped_files)):
                cmd = 'sbatch -p rondror -t 0:20:00 -o {} --wrap="$SCHRODINGER/run python3 group.py group {} {} {} ' \
                      '--protein {} --target {} --start {} --index {}"'
                out_file_name = 'subsample_{}_{}_{}_{}.out'.format(protein, target, start, i)
                os.system(
                    cmd.format(os.path.join(args.run_path, out_file_name), args.docked_prot_file, args.run_path,
                               args.raw_root, protein, target, start, i))
                counter += 1

        print(counter)

    elif args.task == 'group':
        pair = '{}-to-{}'.format(args.target, args.start)
        protein_path = os.path.join(args.raw_root, args.protein)
        pair_path = os.path.join(protein_path, pair)
        grid_size = get_grid_size(pair_path, args.target, args.start)
        pose_path = os.path.join(pair_path, 'exhaustive_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size))
        correct_path = os.path.join(pose_path, 'correct_after_simple_filter')
        if not os.path.exists(correct_path):
            os.mkdir(correct_path)
        subsample_path = os.path.join(pose_path, 'subsample_incorrect')
        if not os.path.exists(subsample_path):
            os.mkdir(subsample_path)

        prefix = 'exhaustive_search_poses_'
        suffix = '.csv'
        files = [f for f in os.listdir(pose_path) if f[:len(prefix)] == prefix]
        grouped_files = group_files(args.n, files)

        for file in grouped_files[args.index]:
            start_time = time.time()
            name = file[len(prefix):-len(suffix)]
            df = pd.read_csv(os.path.join(pose_path, file))
            correct_df = df[df['rmsd'] <= args.rmsd_cutoff]
            correct_df.to_csv(os.path.join(correct_path, 'correct_{}.csv'.format(name)))
            incorrect_df = df[df['rmsd'] > args.rmsd_cutoff]
            indices = [i for i in range(len(incorrect_df))]
            random.shuffle(indices)
            indices = indices[:300]
            conformers = sorted(indices)
            outfile = open(os.path.join(subsample_path, 'index_{}.pkl'.format(name)), 'wb')
            pickle.dump(conformers, outfile)
            print(time.time() - start_time)

    elif args.task == 'all_combine':
        pairs = get_prots(args.docked_prot_file)
        random.shuffle(pairs)
        for protein, target, start in pairs[:5]:
            if protein == 'Q86WV6':
                continue
            cmd = 'sbatch -p rondror -t 0:20:00 -o {} --wrap="$SCHRODINGER/run python3 group.py group_combine ' \
                  '{} {} {} --protein {} --target {} --start {}"'
            out_file_name = 'subsample_combine_{}_{}_{}.out'.format(protein, target, start)
            os.system(
                cmd.format(os.path.join(args.run_path, out_file_name), args.docked_prot_file, args.run_path,
                           args.raw_root, protein, target, start))

    elif args.task == 'group_combine':
        pair = '{}-to-{}'.format(args.target, args.start)
        protein_path = os.path.join(args.raw_root, args.protein)
        pair_path = os.path.join(protein_path, pair)
        grid_size = get_grid_size(pair_path, args.target, args.start)
        pose_path = os.path.join(pair_path, 'exhaustive_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size))
        correct_path = os.path.join(pose_path, 'correct_after_simple_filter')

        prefix = 'exhaustive_search_poses_'
        suffix = '.csv'
        files = [f for f in os.listdir(pose_path) if f[:len(prefix)] == prefix]
        dfs = []
        for file in files:
            name = file[len(prefix):-len(suffix)]
            correct_file = os.path.join(correct_path, 'correct_{}.csv'.format(name))
            df = pd.read_csv(correct_file)
            dfs.append(df)

        combined_df = pd.concat(dfs)
        combined_df.to_csv(os.path.join(correct_path, 'combined.csv'))


if __name__ == "__main__":
    main()
