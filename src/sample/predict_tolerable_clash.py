"""
The purpose of this code is to first create the raw directory folder and include the following files
starting protein receptor
starting ligand
target ligand
glide pose viewer file

Then the top glide poses are added

Then the decoys are created

It can be run on sherlock using
$ $SCHRODINGER/run python3 systematic_decoy_search.py get_clash_data /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /home/users/sidhikab/lig_clash_score/src/data/decoy_timing_data_with_clash --rotation_search_step_size 5 --grid_size 1 --grid_n 1 --remove_prot_h --prot_pocket_only --clash_filter --clash_cutoff 15 --protein A5F5R2 --target 4x24 --start 4wkb --index 0
"""

import argparse
import random


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
    args = parser.parse_args()

    if args.task == 'all_search':
        process = get_prots(args.docked_prot_file)
        random.shuffle(process)
        


if __name__=="__main__":
    main()