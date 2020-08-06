'''
This protocol can be used to find the amino acid sequence for a set of structures
The result is stored in a pickled 2D array
The 2D array will be used for pairwise alignment

Store outputs in Data/Alignments
Store 1 alignment pickled file per protein

how to run this file:
ml load chemistry
ml load schrodinger
$SCHRODINGER/run python3 sequence_finder.py /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt
'''

import argparse
from schrodinger.structure import StructureReader
import os
import pickle
from tqdm import tqdm

data_root = '/oak/stanford/groups/rondror/projects/ligand-docking/pdbbind_2019/data'
backup_root = '/oak/stanford/groups/rondror/projects/combind/flexibility/pdbbind_2019/data'
save_path = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/'


def get_sequence_from_str(file, chains, protein):
    s = list(StructureReader(file))[0]
    str = ''
    for m in list(s.molecule):
        if len(m.residue) != 1:
            for r in list(m.residue):
                atom = list(r.atom)[0]
                if atom.chain == chains[protein]:
                    str += atom.pdbcode
    return str

'''
Get the amino acid sequence
:param file: .mae file for the structure
:return: the amino acid string for all amino acids in chain A
'''
def process(protein, seqs, chains, ligand, error_count, chain_error):
    protein_folder = os.path.join(data_root, protein + '/structures/aligned')
    if protein not in seqs:
        seqs[protein] = {}
    if ligand not in seqs[protein]:
        seq = ''
        primary_file = os.path.join(protein_folder, ligand + '_prot.mae')
        if os.path.exists(primary_file):
            seq = get_sequence_from_str(primary_file, chains, protein)
        else:
            print("File does not exist:", ligand + '_prot.mae')
            error_count += 1
        if seq != '':
            seqs[protein][ligand] = seq
        else:
            chain_error += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    args = parser.parse_args()

    infile = open(os.path.join(save_path, 'chains.pkl'), 'rb')
    chains = pickle.load(infile)
    infile.close()

    seqs = {}
    error_count = 0
    chain_error = 0
    with open(args.docked_prot_file) as fp:
        for line in tqdm(fp, desc='protein file'):
            if line[0] == '#': continue
            protein, target, start = line.strip().split()
            process(protein, seqs, chains, start, error_count, chain_error)
            process(protein, seqs, chains, target, error_count, chain_error)

    print(error_count, "files not found")
    print(chain_error, "empty sequences")

    with open(save_path + 'sequences.pkl', 'wb') as f:
        pickle.dump(seqs, f)

if __name__ == '__main__':
    main()
