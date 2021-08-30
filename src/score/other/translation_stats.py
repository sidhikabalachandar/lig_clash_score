"""
The purpose of this code is to find the mean and stdev of the set of all rmsds of all outputted glide poses

It can be run on sherlock using
/home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python translation_stats.py /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /oak/stanford/groups/rondror/projects/ligand-docking/pdbbind_2019/data /home/users/sidhikab/lig_clash_score/reports/figures/lig_translation.png
"""

import pandas as pd
import statistics
import seaborn as sns
from tqdm import tqdm
import argparse

CUTOFF = 40
MAX_POSES = 100

def get_prots(docked_prot_file, raw_root):
    """
    gets rmsds of the docking runs of all protein, target ligands, and starting ligands in the index file
    :param docked_prot_file: (string) file listing proteins to process
    :return: process (list) list of all protein, target ligands, and starting ligands to process
    """
    rmsds = {}
    with open(docked_prot_file) as fp:
        for line in tqdm(fp, desc='going through protein, target, start groups'):
            if line[0] == '#': continue
            protein, target, start = line.strip().split()
            rmsd_file = '{}/{}/docking/sp_es4/{}-to-{}/rmsd.csv'.format(raw_root, protein, target, start)
            df = pd.read_csv(rmsd_file)
            if protein not in rmsds:
                rmsds[protein] = {}
            if start not in rmsds[protein]:
                rmsds[protein][start] = {}
            rmsds[protein][start][target] = df['RMSD'][:MAX_POSES]

    return rmsds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('save_path', type=str, help='path to saved figure')
    args = parser.parse_args()

    rmsds = get_prots(args.docked_prot_file, args.raw_root)

    rmsds_ls = []
    for protein in rmsds:
        for start in rmsds[protein]:
            for target in rmsds[protein][start]:
                rmsds_ls.extend(rmsds[protein][start][target])

    print("Average rmsd is:", statistics.mean(rmsds_ls))
    print("Stdev of rmsd is:", statistics.stdev(rmsds_ls))
    print('Max is:', max(rmsds_ls))

    ax = sns.distplot(rmsds_ls)
    fig = ax.get_figure()
    fig.savefig(args.save_path)


if __name__=="__main__":
    main()