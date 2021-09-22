"""
The purpose of this code is to create conformers

It can be run on sherlock using
$ $SCHRODINGER/run python3 clash_distribution.py /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw --cutoff 0
"""

import argparse
import os
import random
import schrodinger.structure as structure
import schrodinger.structutils.interactions.steric_clash as steric_clash
import seaborn as sns
import pickle
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import sys
sys.path.insert(1, '../util')
from util import *
from schrod_replacement_util import *


# MAIN TASK FUNCTIONS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('--cutoff', type=int, default=15, help='clash cutoff between target protein and '
                                                                           'ligand pose')
    args = parser.parse_args()

    random.seed(0)

    process = get_prots(args.docked_prot_file)

    volumes = []

    print(len(process))

    if not os.path.exists('clash_custom.pkl'):

        for protein, target, start in process[:500]:
            pair = '{}-to-{}'.format(target, start)
            protein_path = os.path.join(args.raw_root, protein)
            pair_path = os.path.join(protein_path, pair)
            target_lig_file = os.path.join(pair_path, 'ligand_poses', '{}_lig0.mae'.format(target))
            target_lig = list(structure.StructureReader(target_lig_file))[0]
            start_prot_file = os.path.join(pair_path, '{}_prot.mae'.format(start))
            start_prot = list(structure.StructureReader(start_prot_file))[0]

            # volume_docking = steric_clash.clash_volume(start_prot, struc2=target_lig)
            grid, origin = get_grid(start_prot)
            volume_docking = get_clash(target_lig, grid, origin)
            volumes.append(volume_docking)

        outfile = open('clash_custom.pkl', 'wb')
        pickle.dump(volumes, outfile)
        outfile.close()

        fig, ax = plt.subplots()
        sns.distplot(volumes, hist=True)
        plt.title('Clash Distribution')
        plt.xlabel('clash volume')
        plt.ylabel('frequency')
        ax.legend()
        fig.savefig('clash_custom.png')

    else:
        infile = open('clash_custom.pkl', 'rb')
        volumes = pickle.load(infile)
        infile.close()

        x = []
        y = []

        for i in range(max(volumes) + 1):
            fine = [i for i in volumes if i == args.cutoff]
            x.append(i)
            y.append(len(fine) / len(volumes))

        fig, ax = plt.subplots()
        ax.bar(x, y)
        fig.savefig('clash_custom_prob.png')


if __name__ == "__main__":
    main()
