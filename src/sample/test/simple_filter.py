

"""
The purpose of this code is to create conformers

It can be run on sherlock using
$ $SCHRODINGER/run python3 simple_filter.py all /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits/search_test_incorrect_glide_index.txt /home/users/sidhikab/lig_clash_score/src/sample/test/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw --protein C8B467 --target 5ult --start 5uov
"""

import argparse
import os
import random
import pandas as pd
import time
import sys
import schrodinger.structure as structure
sys.path.insert(1, '../util')
from util import *
from prot_util import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either align or search')
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('raw_root', type=str, help='directory where script and output files will be written')
    parser.add_argument('--residue_cutoff', type=int, default=0, help='name of pose group subdir')
    parser.add_argument('--rmsd_cutoff', type=float, default=2.5, help='name of pose group subdir')
    parser.add_argument('--num_conformers', type=int, default=300, help='maximum number of conformers considered')
    parser.add_argument('--grid_n', type=int, default=75, help='number of grid_points processed in each job')
    parser.add_argument('--conformer_n', type=int, default=10, help='number of conformers processed in each job')
    parser.add_argument('--start_clash_cutoff', type=int, default=1, help='clash cutoff between start protein and '
                                                                          'ligand pose')
    parser.add_argument('--protein', type=str, default='', help='name of protein')
    parser.add_argument('--target', type=str, default='', help='name of target ligand')
    parser.add_argument('--start', type=str, default='', help='name of start ligand')
    args = parser.parse_args()
    random.seed(0)

    if args.task == 'all':
        pairs = get_prots(args.docked_prot_file)
        random.shuffle(pairs)
        for protein, target, start in pairs[5:10]:
            cmd = 'sbatch -p rondror -t 1:00:00 -o {} --wrap="$SCHRODINGER/run python3 simple_filter.py group {} {} ' \
                  '{} --protein {} --target {} --start {}"'
            out_file_name = 'filter_{}_{}_{}.out'.format(protein, target, start)
            os.system(
                cmd.format(os.path.join(args.run_path, out_file_name), args.docked_prot_file, args.run_path,
                           args.raw_root, protein, target, start))

    if args.task == 'group':
        pair = '{}-to-{}'.format(args.target, args.start)
        protein_path = os.path.join(args.raw_root, args.protein)
        pair_path = os.path.join(protein_path, pair)
        conformer_file = os.path.join(pair_path, "aligned_to_start_with_hydrogen_conformers.mae")
        conformers = list(structure.StructureReader(conformer_file))[:args.num_conformers]
        conformer_indices = [i for i in range(len(conformers))]
        grouped_conformer_indices = group_files(args.conformer_n, conformer_indices)

        grid_size = get_grid_size(pair_path, args.target, args.start)
        grouped_grid_locs = group_grid(args.grid_n, grid_size, 2)

        group_name = 'test_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size)
        pose_path = os.path.join(pair_path, group_name)

        for i in range(len(grouped_grid_locs)):
            for j in range(len(grouped_conformer_indices)):
                data_dict = {'start_clash_cutoff': [], 'num_poses_searched': [], 'num_correct': [],
                             'num_after_simple_filter': [], 'num_correct_after_simple_filter': []}
                info_file = os.path.join(pose_path, 'exhaustive_search_info_{}_{}.csv'.format(i, j))
                info_df = pd.read_csv(info_file)
                pose_file = os.path.join(pose_path, 'exhaustive_search_poses_{}_{}.csv'.format(i, j))
                pose_df = pd.read_csv(pose_file)

                # cutoff = 1
                data_dict['start_clash_cutoff'].append(1)
                data_dict['num_poses_searched'].append(info_df['num_poses_searched'].iloc[0])
                data_dict['num_correct'].append(info_df['num_correct'].iloc[0])

                filtered_df = pose_df[pose_df['start_clash'] < args.start_clash_cutoff]
                data_dict['num_after_simple_filter'].append(len(filtered_df))
                data_dict['num_correct_after_simple_filter'].append(
                    len(filtered_df[filtered_df['rmsd'] < args.rmsd_cutoff]))

                df = pd.DataFrame.from_dict(data_dict)
                df.to_csv(info_file)


if __name__ == "__main__":
    main()

