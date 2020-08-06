"""
The purpose of this code is to create the split files
It can be run on sherlock using
$ /home/groups/rondror/software/sidhikab/miniconda/envs/geometric/bin/python dock_to_dict.py /home/users/sidhikab/flexibility_project/atom3d/Data/refined_random.txt /home/users/sidhikab/plep/index/INDEX_refined_name.2019
"""

import argparse
import matplotlib.pyplot as plt

data_path = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('all_prot_file', type=str, help='file listing proteins to process')
    args = parser.parse_args()

    proteins = {}
    with open(args.docked_prot_file) as fp:
        for line in fp:
            if line[0] == '#': continue
            protein, _, _ = line.strip().split()
            if protein not in proteins:
                proteins[protein] = 0
            proteins[protein] += 1

    print(len(proteins))
    print("Average", sum(proteins.values()) / len(proteins))

    counts = {}
    for i in proteins.values():
        if i not in counts:
            counts[i] = 0
        counts[i] += 1

    fig, ax = plt.subplots()
    ax.bar(counts.keys(), counts.values())
    ax.set(xlabel='Num Docking Runs', ylabel='Num Proteins',
           title='Histogram of number of ligands for docked proteins from PDBBind')
    plt.savefig('/home/users/sidhikab/flexibility_project/atom3d/reports/figures/docked_ligand_count.png')

    proteins = {}
    with open(args.all_prot_file) as fp:
        for line in fp:
            if line[0] == '#': continue
            pdb, _, uniprot, *_ = line.strip().split()
            if uniprot == '------': continue

            if uniprot not in proteins:
                proteins[uniprot] = 0
            proteins[uniprot] += 1

    for i in proteins.items():
        proteins[i[0]] = proteins[i[0]] * proteins[i[0]] - proteins[i[0]]

    print(len(proteins))
    print("Average", sum(proteins.values()) / len(proteins))

    counts = {}
    for i in proteins.values():
        if i < 50:
            if i not in counts:
                counts[i] = 0
            counts[i] += 1

    fig, ax = plt.subplots()
    ax.bar(counts.keys(), counts.values())
    ax.set(xlabel='Num Docking Runs', ylabel='Num Proteins',
           title='Histogram of number of ligands for all 2019 refined proteins from PDBBind')
    plt.savefig('/home/users/sidhikab/flexibility_project/atom3d/reports/figures/all_ligand_count.png')

if __name__=="__main__":
    main()