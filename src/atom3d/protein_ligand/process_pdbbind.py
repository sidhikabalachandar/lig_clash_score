"""
The purpose of this code is to process the pdb files
It can be run on sherlock using
$ /home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python process_pdbbind.py all /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt --out_dir
$ /home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python process_pdbbind.py group /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt --group <index> --out_dir /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/processed
$ /home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python process_pdbbind.py check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt --out_dir /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/processed
$ /home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python process_pdbbind.py MAPK14 /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt --out_dir /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/processed
"""

#!/usr/bin/env python
# coding: utf-8


import os
import scipy.spatial
import sys
sys.path.append('..')
from util import datatypes as dt
from util import file as fi
from rdkit import Chem
import Bio.PDB
from Bio.PDB.PDBIO import Select
from tqdm import tqdm
import argparse

NUM_ITEMS = 5
run_path = '/home/users/sidhikab/flexibility_project/atom3d/src/atom3d/protein_ligand/run'

def get_all_files(input_dir, group, proteins, ending):
    pdb_files = []
    pdb_file_endings = []
    for protein in group:
        protein_path = os.path.join(input_dir, protein)
        for pair in proteins[protein]:
            for file in fi.find_files(os.path.join(protein_path, pair), ending):
                if file.split('/')[-1] not in pdb_file_endings:
                    pdb_files.append(file)
                    pdb_file_endings.append(file.split('/')[-1])

        return pdb_files

def get_pdb_files(input_dir, protein, pdb, proteins, ending):
    pdb_files = []
    pdb_file_endings = []
    protein_path = os.path.join(input_dir, protein)
    for pair in proteins[protein]:
        if pdb in pair.split('-to-'):
            for file in fi.find_files(os.path.join(protein_path, pair), ending):
                if file.split('/')[-1] not in pdb_file_endings and fi.get_pdb_code(file) == pdb.lower():
                    pdb_files.append(file)
                    pdb_file_endings.append(file.split('/')[-1])

    return pdb_files


def get_prots(fname, out_path):
    unfinished_proteins = {}
    all_proteins = {}
    ligands = []
    dup_target = []
    with open(fname) as fp:
        for line in tqdm(fp, desc='files'):
            if line[0] == '#': continue
            protein, target, start = line.strip().split()
            if target not in ligands:
                ligands.append(target)
            else:
                dup_target.append(target)
            if start not in ligands:
                ligands.append(start)
            if protein not in all_proteins:
                all_proteins[protein] = []
            all_proteins[protein].append('{}-to-{}'.format(target, start))
            if not os.path.exists(os.path.join(out_path, start)):
                if protein not in unfinished_proteins:
                    unfinished_proteins[protein] = []
                if start not in unfinished_proteins[protein]:
                    unfinished_proteins[protein].append(start)
            if not os.path.exists(os.path.join(out_path, target)):
                if protein not in unfinished_proteins:
                    unfinished_proteins[protein] = []
                if target not in unfinished_proteins[protein]:
                    unfinished_proteins[protein].append(target)

    return unfinished_proteins, all_proteins, ligands, dup_target

def check_extra(ligands, out_path):
    extra = []
    for ligand in os.listdir(out_path):
        if ligand not in ligands:
            extra.append(ligand)

    return extra

def get_ligand(ligfile):
    """
    Read ligand from PDB dataset into RDKit Mol. Assumes input is sdf format.
    """
    lig=Chem.SDMolSupplier(ligfile)[0]
    # Many SDF in PDBBind do not parse correctly. If SDF fails, try loading the mol2 file instead
    if lig is None:
        print('trying mol2...')
        lig=Chem.MolFromMol2File(ligfile[:-4] + '.mol2')
    if lig is None:
        print('failed')
        return None
    lig = Chem.RemoveHs(lig)
    return lig


def get_pocket_res(protein, ligand, dist):
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
    prot_atoms = [a for a in protein.get_atoms()]
    prot_coords = [atom.get_coord() for atom in prot_atoms]
        
    # get ligand coordinates
    lig_coords = []
    for i in range(0, ligand.GetNumAtoms()):
        pos = ligand.GetConformer().GetAtomPosition(i)
        lig_coords.append([pos.x, pos.y, pos.z])
        
    kd_tree = scipy.spatial.KDTree(prot_coords)
    key_pts = kd_tree.query_ball_point(lig_coords, r=dist, p=2.0)
    key_pts = set([k for l in key_pts for k in l])
    
    key_residues = set() 
    for i in key_pts:
        atom = prot_atoms[i]
        res = atom.get_parent()
        if res.get_resname() == 'HOH':
            continue
        key_residues.add(res)
    return key_residues


class PocketSelect(Select):
    """
    Selection class for subsetting protein to key binding residues
    """
    def __init__(self, reslist):
        self.reslist = reslist
    def accept_residue(self, residue):
        if residue in self.reslist:
            return True
        else:
            return False


def process_files(input_dir, protein, pdb, proteins):
    """
    Process all protein (pdb) and ligand (sdf) files in input directory.
    Args
        input dir (str): directory containing PDBBind data
    Returns
        structure_dict (dict): dictionary containing each structure, keyed by PDB code. Each PDB is a dict containing protein in Biopython format and ligand in RDKit Mol format
    """
    structure_dict = {}
    pdb_files = get_pdb_files(input_dir, protein, pdb, proteins, 'pdb')\

    for f in tqdm(pdb_files, desc='pdb files'):
        pdb_id = fi.get_pdb_code(f)
        if pdb_id not in structure_dict:
            structure_dict[pdb_id] = {}
        if '_prot' in f:
            prot = dt.read_any(f)
            if 'protein' not in structure_dict[pdb_id]:
                structure_dict[pdb_id]['protein'] = prot

    lig_files = get_pdb_files(input_dir, protein, pdb, proteins, 'sdf')
    for f in tqdm(lig_files, desc='ligand files'):
        pdb_id = fi.get_pdb_code(f)
        if fi.get_pdb_name(f) not in structure_dict[pdb_id]:
            structure_dict[pdb_id][fi.get_pdb_name(f)] = get_ligand(f)
    
    return structure_dict



def write_files(pdbid, protein, ligand, pocket, out_path, index):
    """
    Writes cleaned structure files for protein, ligand, and pocket.
    """
    # write protein to mmCIF file
    io = Bio.PDB.MMCIFIO()
    io.set_structure(protein)
    protein_path = os.path.join(out_path, pdbid)
    if not os.path.exists(protein_path):
        os.mkdir(protein_path)
    if not os.path.exists(f"{pdbid}_protein.mmcif"):
        io.save(os.path.join(protein_path, f"{pdbid}_protein.mmcif"))
    
    # write pocket to mmCIF file
    io.save(os.path.join(protein_path, f"{pdbid}_pocket{index}.mmcif"), PocketSelect(pocket))
    
    # write ligand to file
    writer = Chem.SDWriter(os.path.join(protein_path, f"{pdbid}_ligand{index}.sdf"))
    writer.write(ligand)


def produce_cleaned_dataset(structure_dict, out_path, dist):
    """
    Generate cleaned dataset in out_path, given dictionary of structures processed by process_files.
    """
    for pdb, data in tqdm(structure_dict.items(), desc='writing to files'):
        protein = structure_dict[pdb]['protein']
        for lig_name in structure_dict[pdb]:
            if lig_name == 'protein':
                continue
            ligand = structure_dict[pdb][lig_name]
            # check for failed ligand (due to bad structure file)
            if ligand is None:
                continue
            pocket_res = get_pocket_res(protein, ligand, dist)
            write_files(pdb, protein, ligand, pocket_res, out_path, lig_name[3:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either all or group')
    parser.add_argument('data_dir', type=str, help='directory where PDBBind is located')
    parser.add_argument('prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('--protein', type=str, default='None', help='if type is group, argument indicates group index')
    parser.add_argument('--pdb', type=str, default='None', help='if type is group, argument indicates group index')
    parser.add_argument('--dist', type=float, default=6.0, help='distance cutoff for defining pocket')
    parser.add_argument('--out_dir', type=str, default=os.getcwd(), help='directory to place cleaned dataset')
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        raise Exception('Path not found. Please enter valid path to PDBBind dataset.')
    
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    if args.task == 'all':
        unfinished_proteins, all_proteins, ligands, dup_target = get_prots(args.prot_file, args.out_dir)
        prot_names = sorted(list(unfinished_proteins.keys()))

        grouped_files = []
        n = 1

        for i in range(0, len(prot_names), n):
            grouped_files += [prot_names[i: i + n]]

        counter = 0
        for protein in unfinished_proteins:
            for code in unfinished_proteins[protein]:
                cmd = 'sbatch -p rondror -t 5:00:00 -o {} --wrap=' \
                      '"/home/groups/rondror/software/sidhikab/miniconda/envs/geometric/bin/python ' \
                      'process_pdbbind.py group /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw ' \
                      '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt ' \
                      '--protein {} --pdb {} --out_dir /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/processed"'
                os.system(cmd.format(os.path.join(run_path, 'process_{}.out'.format(code)), protein, code))
                # print(cmd.format(os.path.join(run_path, 'process_{}.out'.format(code)), protein, code))
                counter += 1

    if args.task == 'group':
        unfinished_proteins, all_proteins, ligands, dup_target = get_prots(args.prot_file, args.out_dir)
        prot_names = sorted(list(unfinished_proteins.keys()))

        grouped_files = []
        n = 1

        for i in range(0, len(prot_names), n):
            grouped_files += [prot_names[i: i + n]]

        print(args.protein, args.pdb)
        structures = process_files(args.data_dir, args.protein, args.pdb, all_proteins)
        produce_cleaned_dataset(structures, args.out_dir, args.dist)

    if args.task == 'check':
        unfinished_ligs = []
        with open(args.prot_file) as fp:
            for line in tqdm(fp, desc='files'):
                if line[0] == '#': continue
                protein, target, start = line.strip().split()
                start_dir = os.path.join(args.out_dir, start)
                protein_file = '{}_protein.mmcif'.format(start)
                try:
                    dt.bp_to_df(dt.read_any(os.path.join(start_dir, protein_file)))
                except Exception as e:
                    unfinished_ligs.append((protein, start))

        print(len(unfinished_ligs))
        # print(unfinished_ligs)

        # for protein, ligand in unfinished_ligs:
        #         cmd = 'sbatch -p rondror -t 5:00:00 -o {} --wrap=' \
        #               '"/home/groups/rondror/software/sidhikab/miniconda/envs/atom3d/bin/python ' \
        #               'process_pdbbind.py group /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw ' \
        #               '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt ' \
        #               '--protein {} --pdb {} --out_dir /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/processed"'
        #         os.system(cmd.format(os.path.join(run_path, 'process_{}.out'.format(ligand)), protein, ligand))
                # print(cmd.format(os.path.join(run_path, 'process_{}.out'.format(code)), protein, code))
    if args.task == 'MAPK14':
        protein = 'MAPK14'
        ligs = ['4F9Y']
        for pdb in ligs:
            structures = process_files(args.data_dir, protein,  pdb, {'MAPK14': ['4F9Y-to-3D83']})
            produce_cleaned_dataset(structures, args.out_dir, args.dist)

if __name__ == "__main__":
    main()

