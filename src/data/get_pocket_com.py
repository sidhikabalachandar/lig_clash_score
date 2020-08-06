"""
The purpose of this code is to set the train, val, and test data sets
It can be run on sherlock using
ml load chemistry
ml load schrodinger
$ $SCHRODINGER/run python3 get_pocket_com.py
"""

from tqdm import tqdm
import pickle
import schrodinger.structutils.analyze as analyze
from schrodinger.structure import StructureReader
import os
import scipy.spatial

prot_file = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random_with_unaligned.txt'
data_root = '/oak/stanford/groups/rondror/projects/ligand-docking/pdbbind_2019/data'
DIST = 6.0


def get_volume(structs):
    x_dim = max(structs[:, 0]) - min(structs[:, 0])
    y_dim = max(structs[:, 1]) - min(structs[:, 1])
    z_dim = max(structs[:, 2]) - min(structs[:, 2])
    return x_dim * y_dim * z_dim


def get_pocket_res(protein, ligand):
    """
    Given a co-crystallized protein and ligand, extract residues within specified distance of ligand.

    Args:
        protein (Biopython Structure object): receptor protein
        ligand (RDKit Mol object): co-crystallized ligand
        dist (float): distance cutoff for defining binding site

    Returns:
        key_residues (set of Biopython Residue objects): set of key binding site residues
    """
    # get protein coordinates
    prot_atoms = protein.getAtomIndices()
    prot_coords = protein.getXYZ()

    # get ligand coordinates
    lig_coords = ligand.getXYZ()

    kd_tree = scipy.spatial.KDTree(prot_coords)
    key_pts = kd_tree.query_ball_point(lig_coords, r=DIST, p=2.0)
    key_pts = set([k for l in key_pts for k in l])
    return analyze.center_of_mass(protein, list(key_pts.intersection(prot_atoms)))

def main():
    coms = {}
    with open(prot_file) as fp:
        for line in tqdm(fp, desc='protein file'):
            if line[0] == '#': continue
            protein, target, start = line.strip().split()

            if protein not in coms:
                coms[protein] = {}

            if start not in coms[protein]:
                start_receptor_file = os.path.join(data_root, '{}/structures/aligned/{}_prot.mae'.format(protein, start))
                start_ligand_file = os.path.join(data_root, '{}/structures/aligned/{}_lig.mae'.format(protein, start))
                start_struct = list(StructureReader(start_receptor_file))[0]
                start_lig = list(StructureReader(start_ligand_file))[0]
                # print(protein, start)
                coms[protein][start] = get_pocket_res(start_struct, start_lig)

            if target not in coms[protein]:
                target_receptor_file = os.path.join(data_root, '{}/structures/aligned/{}_prot.mae'.format(protein, target))
                target_ligand_file = os.path.join(data_root, '{}/structures/aligned/{}_lig.mae'.format(protein, target))
                target_struct = list(StructureReader(target_receptor_file))[0]
                target_lig = list(StructureReader(target_ligand_file))[0]
                # print(protein, target)
                get_pocket_res(target_struct, target_lig)


    with open('pocket_com.pkl', 'wb') as f:
        pickle.dump(coms, f)

if __name__=="__main__":
    main()