"""
The purpose of this code is to set the train, val, and test data sets
It can be run on sherlock using
/home/groups/rondror/software/sidhikab/miniconda/envs/geometric/bin/python translation_stats.py
"""

import os
import pandas as pd
import statistics
import seaborn as sns
from tqdm import tqdm

CUTOFF = 40
prot_file = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt'
save_root = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw'
data_root = '/oak/stanford/groups/rondror/projects/ligand-docking/pdbbind_2019/data'

def main():
    rmsds = {}

    with open(prot_file) as fp:
        for line in tqdm(fp, desc='protein file'):
            if line[0] == '#': continue
            protein, target, start = line.strip().split()
            pv_file = os.path.join(save_root,
                                   '{}/{}-to-{}/{}-to-{}_pv.maegz'.format(protein, target, start, target, start))
            if os.path.exists(pv_file):
                rmsd_file = '{}/{}/docking/sp_es4/{}-to-{}/rmsd.csv'.format(data_root, protein, target, start)
                df = pd.read_csv(rmsd_file)
                if protein not in rmsds:
                    rmsds[protein] = {}
                if start not in rmsds[protein]:
                    rmsds[protein][start] = {}
                rmsds[protein][start][target] = df['RMSD']

    rmsds_ls = []
    for protein in rmsds:
        for start in rmsds[protein]:
            for target in rmsds[protein][start]:
                rmsds_ls.extend(rmsds[protein][start][target])

    print("Average rmsd is:", statistics.mean(rmsds_ls))
    print("Stdev of rmsd is:", statistics.stdev(rmsds_ls))

    ax = sns.distplot(rmsds_ls)
    fig = ax.get_figure()
    fig.savefig('/home/users/sidhikab/flexibility_project/atom3d/reports/figures/lig_translation.png')


if __name__=="__main__":
    main()