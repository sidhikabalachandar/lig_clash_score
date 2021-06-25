"""
The purpose of this code is to create conformers

It can be run on sherlock using
$ $SCHRODINGER/run python3 group_correct.py all /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits/search_test_incorrect_glide_index.txt /home/users/sidhikab/lig_clash_score/src/sample/test/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw --protein P00797 --target 3own --start 3d91 --index 0 --n 1
"""

import argparse
import random
import os
import pandas as pd
import schrodinger.structure as structure
from schrodinger.structutils.transform import get_centroid
import numpy as np


def split(filehandler, delimiter=',', row_limit=10000,
          output_name_template='output_%s.csv', output_path='.', keep_headers=True):
    """
    Splits a CSV file into multiple pieces.

    A quick bastardization of the Python CSV library.
    Arguments:
        `row_limit`: The number of rows you want in each output file. 10,000 by default.
        `output_name_template`: A %s-style template for the numbered output files.
        `output_path`: Where to stick the output files.
        `keep_headers`: Whether or not to print the headers in each output file.
    Example usage:

        >> from toolbox import csv_splitter;
        >> csv_splitter.split(open('/home/ben/input.csv', 'r'));

    """
    import csv
    reader = csv.reader(filehandler, delimiter=delimiter)
    current_piece = 1
    current_out_path = os.path.join(
        output_path,
        output_name_template % current_piece
    )
    current_out_writer = csv.writer(open(current_out_path, 'w'), delimiter=delimiter)
    current_limit = row_limit
    if keep_headers:
        headers = next(reader)
        current_out_writer.writerow(headers)
    for i, row in enumerate(reader):
        if i + 1 > current_limit:
            current_piece += 1
            current_limit = row_limit * current_piece
            current_out_path = os.path.join(
                output_path,
                output_name_template % current_piece
            )
            current_out_writer = csv.writer(open(current_out_path, 'w'), delimiter=delimiter)
            if keep_headers:
                current_out_writer.writerow(headers)
        current_out_writer.writerow(row)


def group_files(n, process):
    """
    groups pairs into sublists of size n
    :param n: (int) sublist size
    :param process: (list) list of pairs to process
    :return: grouped_files (list) list of sublists of pairs
    """
    grouped_files = []

    for i in range(0, len(process), n):
        grouped_files += [process[i: i + n]]

    return grouped_files


def get_prots(docked_prot_file):
    """
    gets list of all protein, target ligands, and starting ligands in the index file
    :param docked_prot_file: (string) file listing proteins to process
    :return: process (list) list of all protein, target ligands, and starting ligands to process
    """
    process = []
    with open(docked_prot_file) as fp:
        for line in fp:
            if line[0] == '#':
                continue
            protein, target, start = line.strip().split()
            process.append((protein, target, start))

    return process


def get_grid_size(pair_path, target, start):
    target_lig_file = os.path.join(pair_path, 'ligand_poses', '{}_lig0.mae'.format(target))
    target_lig = list(structure.StructureReader(target_lig_file))[0]
    target_center = get_centroid(target_lig)

    start_lig_file = os.path.join(pair_path, '{}_lig.mae'.format(start))
    start_lig = list(structure.StructureReader(start_lig_file))[0]
    start_center = get_centroid(start_lig)

    dist = np.sqrt((target_center[0] - start_center[0]) ** 2 +
                   (target_center[1] - start_center[1]) ** 2 +
                   (target_center[2] - start_center[2]) ** 2)

    grid_size = int(dist + 1)
    if grid_size % 2 == 1:
        grid_size += 1
    return grid_size


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
    parser.add_argument('--n', type=int, default=100, help='num jobs per ligand')
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
                cmd = 'sbatch -p rondror -t 0:20:00 -o {} --wrap="$SCHRODINGER/run python3 group_poses.py group {} {} {} ' \
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

        prefix = 'exhaustive_search_poses_'
        suffix = '.csv'
        files = [f for f in os.listdir(pose_path) if f[:len(prefix)] == prefix]
        grouped_files = group_files(args.n, files)

        for file in grouped_files[args.index]:
            name = file[len(prefix):-len(suffix)]
            df = pd.read_csv(os.path.join(pose_path, file))
            correct_df = df[df['rmsd'] <= args.rmsd_cutoff]
            correct_df.to_csv(os.path.join(correct_path, 'correct_{}.csv'.format(name)))



if __name__ == "__main__":
    main()
