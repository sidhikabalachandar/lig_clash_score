"""
The purpose of this code is to create conformers

It can be run on sherlock using
$ $SCHRODINGER/run python3 create_conformers.py test group docked_prot_file run_path raw_root --protein C8B467 --target 5ult --start 5uov --index 0 --n 1
"""

import argparse
import os
import random
import schrodinger.structutils.interactions.steric_clash as steric_clash

import sys
sys.path.insert(1, 'util')
from util import *


# MAIN TASK FUNCTIONS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, help='either train or test')
    parser.add_argument('task', type=str, help='either align or search')
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('--index', type=int, default=-1, help='protein-ligand pair group index')
    parser.add_argument('--num_conformers', type=int, default=300, help='maximum number of conformers considered')
    parser.add_argument('--n', type=int, default=1, help='number of protein, target, start groups processed in '
                                                         'group task')
    parser.add_argument('--protein', type=str, default='', help='name of protein')
    parser.add_argument('--target', type=str, default='', help='name of target ligand')
    parser.add_argument('--start', type=str, default='', help='name of start ligand')
    args = parser.parse_args()

    random.seed(0)

    if not os.path.exists(args.run_path):
        os.mkdir(args.run_path)

    process = get_prots(args.docked_prot_file)

    volumes = []

    for protein, target, start in process:
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(args.raw_root, protein)
        pair_path = os.path.join(protein_path, pair)
        target_lig_file = os.path.join(pair_path, 'ligand_poses', '{}_lig0.mae'.format(target))
        target_lig = list(structure.StructureReader(target_lig_file))[0]
        start_prot_file = os.path.join(pair_path, '{}_prot.mae'.format(args.start))
        start_prot = list(structure.StructureReader(start_prot_file))[0]

        volume_docking = steric_clash.clash_volume(start_prot, struc2=target_lig)
        volumes.append(volume_docking)
        break
        

if __name__ == "__main__":
    main()
