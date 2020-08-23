"""
The purpose of this code is to set the train, val, and test data sets
It can be run on sherlock using
ml load chemistry
ml load schrodinger
$ $SCHRODINGER/run python3 center_of_mass.py
"""

from tqdm import tqdm
import pickle
import schrodinger.structutils.analyze as analyze
from schrodinger.structure import StructureReader
import os

prot_file = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt'
data_root = '/oak/stanford/groups/rondror/projects/ligand-docking/pdbbind_2019/data'


def get_volume(structs):
    x_dim = max(structs[:, 0]) - min(structs[:, 0])
    y_dim = max(structs[:, 1]) - min(structs[:, 1])
    z_dim = max(structs[:, 2]) - min(structs[:, 2])
    return x_dim * y_dim * z_dim

def main():
    coms = {}
    with open(prot_file) as fp:
        for line in tqdm(fp, desc='protein file'):
            if line[0] == '#': continue
            protein, target, start = line.strip().split()
            if protein not in coms:
                coms[protein] = {}
            protein_folder = os.path.join(data_root, protein + '/structures/aligned')
            if start not in coms[protein]:
                file = os.path.join(protein_folder, start + '_prot.mae')
                s = list(StructureReader(file))[0]
                coms[protein][start] = analyze.center_of_mass(s)
                file = os.path.join(protein_folder, start + '_prot.mae')
                s = list(StructureReader(file))[0]
                coms[protein][start] = analyze.center_of_mass(s)

    with open('com_after_aligned.pkl', 'wb') as f:
        pickle.dump(coms, f)

if __name__=="__main__":
    main()