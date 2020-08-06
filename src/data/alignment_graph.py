"""
The purpose of this code is to set the train, val, and test data sets
It can be run on sherlock using
/home/groups/rondror/software/sidhikab/miniconda/envs/geometric/bin/python alignment_graph.py
"""

import pickle
import seaborn as sns
import numpy as np

prot_file = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random_with_unaligned.txt'
data_root = '/oak/stanford/groups/rondror/projects/ligand-docking/pdbbind_2019/data'
com_dict = '/home/users/sidhikab//flexibility_project/atom3d/src/data/pocket_com.pkl'


def get_volume(structs):
    x_dim = max(structs[:, 0]) - min(structs[:, 0])
    y_dim = max(structs[:, 1]) - min(structs[:, 1])
    z_dim = max(structs[:, 2]) - min(structs[:, 2])
    return x_dim * y_dim * z_dim

def main():
    infile = open(com_dict, 'rb')
    coms = pickle.load(infile)
    infile.close()

    volumes = {}
    for protein in coms:
        structs = np.zeros((len(coms[protein]), 3))
        for i, ligand in enumerate(coms[protein]):
            structs[i] = coms[protein][ligand]
        volumes[protein] = get_volume(structs)

    sorted_volumes = sorted(volumes.items(), key=lambda x: x[1], reverse=True)

    ax = sns.distplot([i[1] for i in sorted_volumes])
    fig = ax.get_figure()
    fig.savefig('/home/users/sidhikab/flexibility_project/atom3d/reports/figures/pocket_aligned.png')

    for i in range(5):
        print(sorted_volumes[i])
        for ligand in coms[sorted_volumes[i][0]]:
            print(ligand, coms[sorted_volumes[i][0]][ligand])

if __name__=="__main__":
    main()