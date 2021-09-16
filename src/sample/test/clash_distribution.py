"""
The purpose of this code is to create conformers

It can be run on sherlock using
$ $SCHRODINGER/run python3 clash_distribution.py /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
"""

import argparse
import os
import random
import schrodinger.structure as structure
import schrodinger.structutils.interactions.steric_clash as steric_clash
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import sys
sys.path.insert(1, '../util')
from util import *


# MAIN TASK FUNCTIONS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    args = parser.parse_args()

    random.seed(0)

    process = get_prots(args.docked_prot_file)

    volumes = []

    print(len(process))

    for protein, target, start in process[:500]:
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(args.raw_root, protein)
        pair_path = os.path.join(protein_path, pair)
        target_lig_file = os.path.join(pair_path, 'ligand_poses', '{}_lig0.mae'.format(target))
        target_lig = list(structure.StructureReader(target_lig_file))[0]
        start_prot_file = os.path.join(pair_path, '{}_prot.mae'.format(start))
        start_prot = list(structure.StructureReader(start_prot_file))[0]

        volume_docking = steric_clash.clash_volume(start_prot, struc2=target_lig)
        volumes.append(volume_docking)

    outfile = open('clash.pkl', 'wb')
    pickle.dump(volumes, outfile)
    outfile.close()

    fig, ax = plt.subplots()
    sns.distplot(volumes, hist=True)
    plt.title('Clash Distribution')
    plt.xlabel('clash volume')
    plt.ylabel('frequency')
    ax.legend()
    fig.savefig('clash.png')


if __name__ == "__main__":
    main()
