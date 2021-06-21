"""
The purpose of this code is to create conformers

It can be run on sherlock using
$ $SCHRODINGER/run python3 group_poses.py size_correct /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits/search_test_incorrect_glide_index.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw --group_name exhaustive_grid_6_2_rotation_0_360_20_rmsd_2.5 --protein P00797 --target 3own --start 3d91 --index 0
"""

import argparse
import random
import os
import pandas as pd
import pickle


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either align or search')
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('raw_root', type=str, help='directory where script and output files will be written')
    parser.add_argument('--group_name', type=str, default='', help='name of pose group subdir')
    parser.add_argument('--rmsd_cutoff', type=int, default=2.5,
                        help='rmsd accuracy cutoff between predicted ligand pose '
                             'and true ligand pose')
    parser.add_argument('--protein', type=str, default='', help='name of protein')
    parser.add_argument('--target', type=str, default='', help='name of target ligand')
    parser.add_argument('--start', type=str, default='', help='name of start ligand')
    parser.add_argument('--n', type=int, default=20, help='num jobs per ligand')
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
            pose_path = os.path.join(pair_path, args.group_name)
            prefix = 'exhaustive_search_poses_'
            files = [f for f in os.listdir(pose_path) if f[:len(prefix)] == prefix]
            grouped_files = group_files(args.n, files)

            for i in range(len(grouped_files)):
                cmd = 'sbatch -p rondror -t 0:20:00 -o {} --wrap="$SCHRODINGER/run python3 group_poses.py group {} {} {} ' \
                      '--group_name {} --protein {} --target {} --start {} --index {}"'
                out_file_name = 'subsample_{}_{}_{}_{}.out'.format(protein, target, start, i)
                os.system(
                    cmd.format(os.path.join(args.run_path, out_file_name), args.docked_prot_file, args.run_path,
                               args.raw_root, args.group_name, protein, target, start, i))
                counter += 1

        print(counter)

    elif args.task == 'group':
        pair = '{}-to-{}'.format(args.target, args.start)
        protein_path = os.path.join(args.raw_root, args.protein)
        pair_path = os.path.join(protein_path, pair)
        pose_path = os.path.join(pair_path, args.group_name)
        subsample_path = os.path.join(pose_path, 'subsample')
        if not os.path.exists(subsample_path):
            os.mkdir(subsample_path)

        prefix = 'exhaustive_search_poses_'
        suffix = '.csv'
        files = [f for f in os.listdir(pose_path) if f[:len(prefix)] == prefix]
        grouped_files = group_files(args.n, files)

        for file in grouped_files[args.index]:
            name = file[len(prefix):-len(suffix)]
            df = pd.read_csv(os.path.join(pose_path, file))
            conformers = [i for i in range(len(df))]
            random.shuffle(conformers)
            conformers = conformers[:len(conformers) // 200]
            conformers = sorted(conformers)
            outfile = open(os.path.join(subsample_path, 'index_{}.pkl'.format(name)), 'wb')
            pickle.dump(conformers, outfile)


    elif args.task == "combine":
        pairs = get_prots(args.docked_prot_file)
        random.shuffle(pairs)
        for protein, target, start in pairs[:5]:
            if protein == 'Q86WV6':
                continue

            pair = '{}-to-{}'.format(target, start)
            protein_path = os.path.join(args.raw_root, protein)
            pair_path = os.path.join(protein_path, pair)
            pose_path = os.path.join(pair_path, args.group_name)
            correct_path = os.path.join(pose_path, 'correct_after_simple_filter')
            dfs = []
            for file in os.listdir(correct_path):
                dfs.append(pd.read_csv(os.path.join(correct_path, file)))

            combined_df = pd.concat(dfs)
            combined_df.to_csv(os.path.join(correct_path, 'combined.csv'))

    elif args.task == "size":
        pairs = get_prots(args.docked_prot_file)
        random.shuffle(pairs)
        count = 0
        max_len = 0
        for protein, target, start in pairs[:5]:
            print(protein, target, start)
            if protein == 'Q86WV6':
                continue

            pair = '{}-to-{}'.format(target, start)
            protein_path = os.path.join(args.raw_root, protein)
            pair_path = os.path.join(protein_path, pair)
            pose_path = os.path.join(pair_path, args.group_name)
            subsample_path = os.path.join(pose_path, 'subsample')

            if protein == 'P03368':
                files = group_files(2, os.listdir(subsample_path))
                for group in files:
                    count += 1
                    conformers = []
                    for file in group:
                        infile = open(os.path.join(subsample_path, file), 'rb')
                        conformers.extend(pickle.load(infile))
                        infile.close()
                    max_len = max(max_len, len(conformers))
            else:
                for file in os.listdir(subsample_path):
                    infile = open(os.path.join(subsample_path, file), 'rb')
                    conformers = pickle.load(infile)
                    max_len = max(max_len, len(conformers))
                    count += 1
                    infile.close()

        print(count)
        print(max_len)

    elif args.task == "remove":
        pairs = get_prots(args.docked_prot_file)
        random.shuffle(pairs)
        count = 0
        max_len = 0
        for protein, target, start in pairs[:5]:
            print(protein, target, start)
            if protein == 'Q86WV6':
                continue

            pair = '{}-to-{}'.format(target, start)
            protein_path = os.path.join(args.raw_root, protein)
            pair_path = os.path.join(protein_path, pair)
            pose_path = os.path.join(pair_path, args.group_name)
            subsample_path = os.path.join(pose_path, 'subsample')

            os.system('rm -rf {}'.format(subsample_path))

        print(count)
        print(max_len)

    elif args.task == "remove_correct":
        pairs = get_prots(args.docked_prot_file)
        random.shuffle(pairs)
        count = 0
        max_len = 0
        for protein, target, start in pairs[:5]:
            print(protein, target, start)
            if protein == 'Q86WV6':
                continue

            pair = '{}-to-{}'.format(target, start)
            protein_path = os.path.join(args.raw_root, protein)
            pair_path = os.path.join(protein_path, pair)
            pose_path = os.path.join(pair_path, args.group_name)
            correct_path = os.path.join(pose_path, 'correct_after_simple_filter')
            clash_path = os.path.join(correct_path, 'clash_data')

            os.system('rm -rf {}'.format(clash_path))

        print(count)
        print(max_len)

    elif args.task == "size_correct":
        pairs = get_prots(args.docked_prot_file)
        random.shuffle(pairs)
        count = 0
        for protein, target, start in pairs[:5]:
            if protein == 'Q86WV6':
                continue

            pair = '{}-to-{}'.format(target, start)
            protein_path = os.path.join(args.raw_root, protein)
            pair_path = os.path.join(protein_path, pair)
            pose_path = os.path.join(pair_path, args.group_name)
            correct_path = os.path.join(pose_path, 'correct_after_simple_filter')
            file = os.path.join(correct_path, 'combined.csv')
            df = pd.read_csv(file)
            print(protein, target, start, len(df))
            count += len(df)

        print(count)



if __name__ == "__main__":
    main()
