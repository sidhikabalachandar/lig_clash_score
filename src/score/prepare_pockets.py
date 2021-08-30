from schrodinger.structure import StructureReader, StructureWriter
import schrodinger.structutils.measure as measure
import os
import sys


def extract_pocket(protein, ligand, cutoff):
    # If cutoff is not specified, return the whole protein structure
    if cutoff:
        close_a_indices = measure.get_atoms_close_to_structure(protein, ligand, cutoff)
        pocket = protein.extract(close_a_indices)
        return pocket
    else:
        return protein


def remove_nonpolar_hydrogens(st):
    removal = []
    for atom in st.atom:
        if (atom.element == 'H'):
            bonded = list(atom.bonded_atoms)
            if (len(bonded) > 0):
                if (bonded[0].element not in ['O', 'N', 'P', 'S']):
                    removal.append(atom.index)
            else:
                removal.append(atom.index)
    st.deleteAtoms(removal)


def process_protein(protein_file, ligand_file, cutoff):
    ligand = next(StructureReader(ligand_file))
    protein = next(StructureReader(protein_file))
    st = extract_pocket(protein, ligand, cutoff)
    remove_nonpolar_hydrogens(st)
    return st


def process_pocket_files(protein_files, ligand_files, names, save_directory, cutoff=8):
    i = 0
    for p, l, name in zip(protein_files, ligand_files, names):
        #try:
        print(name, i)
        i = i + 1
        pocket_name = name + '_pocket.mae'
        pocket_path = os.path.join(save_directory, pocket_name)
        if (not os.path.isfile(pocket_path)):
            pocket = process_protein(p, l, cutoff)
            with StructureWriter(pocket_path) as stwr:
                stwr.append(pocket)
        #except:
        #    print("ERROR!! ", name)