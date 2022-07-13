"""
The purpose of this code is to create the .zip structure files

It can be run on sherlock using
$ ml load chemistry
$ ml load schrodinger
$ $SCHRODINGER/run python3 zipper.py run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
$ $SCHRODINGER/run python3 zipper.py check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
$ $SCHRODINGER/run python3 zipper.py update /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw --new_prot_file /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random2.txt
$ $SCHRODINGER/run python3 zipper.py remove_in_log /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
"""

import os
import argparse
import time
from datetime import timedelta

def get_files(root):
    process = []
    for prot in os.listdir(root):
        for lig_pair in os.listdir(os.path.join(root, prot)):
            process.append((prot, lig_pair))

    return process

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either all, group, or check')
    parser.add_argument('--index', type=int, default=-1, help='protein-ligand pair group index')parser.add_argument('--index', type=int, default=-1, help='protein-ligand pair group index')
    args = parser.parse_args()

    root = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw'
    N = 1

    if args.task == 'all':
        process = get_files(root)
        grouped_files = group_files(N, process)

        for i, group in enumerate(grouped_files):
            os.system('sbatch -p owners -t 00:30:00 -o grid{}.out grid{}_in.sh'.format(i, i))

    if args.task == 'group':
        process = get_files(root)
        grouped_files = group_files(N, process)

        for protein, lig_pair in grouped_files[args.index]:
            starting_time = time.time()
            os.chdir(os.path.join(root, protein))
            os.system('zip {}'.format(lig_pair))
            ending_time = time.time()

            print(timedelta(seconds=ending_time - starting_time))


if __name__ == '__main__':
    main()