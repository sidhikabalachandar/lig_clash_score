'''
This protocol can be used to find the amino acid sequence for a set of structures
The result is stored in a pickled 2D array
The 2D array will be used for pairwise alignment

Store outputs in Data/Alignments
Store 1 alignment pickled file per protein

how to run this file:
ml load chemistry
ml load schrodinger
$SCHRODINGER/run python3 chain_finder.py /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt
'''

import argparse
from schrodinger.structure import StructureReader
import os
import pickle
from tqdm import tqdm

data_root = '/oak/stanford/groups/rondror/projects/ligand-docking/pdbbind_2019/data'
backup_root = '/oak/stanford/groups/rondror/projects/combind/flexibility/pdbbind_2019/data'
save_path = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/'

'''
Get the amino acid sequence
:param file: .mae file for the structure
:return: the amino acid string for all amino acids in chain A
'''
def find_chain(protein, chains, ligand):
    protein_folder = os.path.join(data_root, protein + '/structures/aligned')
    primary_file = os.path.join(protein_folder, ligand + '_prot.mae')
    s = list(StructureReader(primary_file))[0]
    chain_list = []
    for m in list(s.molecule):
        for r in list(m.residue):
            atom = list(r.atom)[0]
            if atom.chain not in chain_list:
                chain_list.append(atom.chain)
    if protein not in chains:
        chains[protein] = chain_list
    else:
        chains[protein] = set.intersection(set(chains[protein]), set(chain_list))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    args = parser.parse_args()

    possible_chains = {}
    with open(args.docked_prot_file) as fp:
        for line in tqdm(fp, desc='protein file'):
            if line[0] == '#': continue
            protein, target, start = line.strip().split()
            find_chain(protein, possible_chains, start)
            find_chain(protein, possible_chains, target)

    final_chains = {}
    error_count = 0
    for protein in tqdm(possible_chains, desc='possible chains'):
        if 'A' in possible_chains[protein]:
            final_chains[protein] = 'A'
        elif len(possible_chains[protein]) != 0:
            final_chains[protein] = min(possible_chains[protein])
        else:
            print(protein)
            error_count += 1


    print(error_count, 'proteins failed')

    with open(save_path + 'chains.pkl', 'wb') as f:
        pickle.dump(final_chains, f)

if __name__ == '__main__':
    main()