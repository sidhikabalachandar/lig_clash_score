"""
The purpose of this code is to obtain the binding pocket files for each pose

It can be run on sherlock using
ml load chemistry
ml load schrodinger
$ $SCHRODINGER/run python3 process_pdbbind.py all /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
$ $SCHRODINGER/run python3 process_pdbbind.py group /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw --index 0
$ $SCHRODINGER/run python3 process_pdbbind.py check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
$ $SCHRODINGER/run python3 process_pdbbind.py remove_pv /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
$ $SCHRODINGER/run python3 process_pdbbind.py remove_pockets /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
$ $SCHRODINGER/run python3 process_pdbbind.py check_remove_pockets /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
$ $SCHRODINGER/run python3 process_pdbbind.py MAPK14 /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt --out_dir /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/processed
"""

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
import schrodinger.structure as structure

def get_prots(docked_prot_file):
    """
    gets list of all protein, target ligands, and starting ligands in the index file
    :param docked_prot_file: (string) file listing proteins to process
    :return: process (list) list of all protein, target ligands, and starting ligands to process
    """
    process = []
    with open(docked_prot_file) as fp:
        for line in fp:
            if line[0] == '#': continue
            protein, target, start = line.strip().split()
            process.append((protein, target, start))

    return process

def group_files(n, process):
    """
    groups pairs into sublists of size n
    :param n: (int) sublist size
    :param process: (list) list of pairs to process
    :return: grouped_files (list) list of sublists of pairs
    """
    grouped_files = []

    for i in range(0, len(process), n):
        grouped_files += [process[i: i + n]]

    return grouped_files

def get_ligand(ligfile):
    """
    Read ligand from PDB dataset into RDKit Mol. Assumes input is sdf format.
    :param ligfile: (string) ligand file
    :return: lig: (RDKit Mol object) co-crystallized ligand
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

def process_files(pair_path):
    """
    Process all protein (pdb) and ligand (sdf) files in input directory.
    Args
        pair_path dir (str): directory containing PDBBind data
    Returns
        structure_dict (dict): dictionary containing each structure, keyed by PDB code. Each PDB is a dict containing protein in Biopython format and ligand in RDKit Mol format
    """
    structure_dict = {}
    pose_path = os.path.join(pair_path, 'ligand_poses')

    # get starting protein structure
    pdb_files = fi.find_files(pair_path, 'pdb')
    for f in tqdm(pdb_files, desc='pdb files'):
        prot = dt.read_any(f)
        structure_dict['protein'] = prot

    # get ligand pose structures
    lig_files = fi.find_files(pose_path, 'sdf')
    for f in tqdm(lig_files, desc='ligand files'):
        structure_dict[fi.get_pdb_name(f)] = get_ligand(f)
    
    return structure_dict

def write_files(protein, pocket, pair_path, target, index):
    """
    Writes cleaned structure files for protein, ligand, and pocket.
    :param protein: (Biopython Structure object) receptor protein
    :param pocket: (set of Biopython Residue objects) set of key binding site residues
    :param pair_path: (string) path to directory for protein, target, ligand group
    :param target: (string) target ligand name
    :param index: (string) index of ligand pose
    :return:
    """
    # write protein to mmCIF file
    io = Bio.PDB.MMCIFIO()
    io.set_structure(protein)
    pocket_path = os.path.join(pair_path, 'pockets')
    if not os.path.exists(pocket_path):
        os.mkdir(pocket_path)
    
    # write pocket to mmCIF file
    io.save(os.path.join(pocket_path, f"{target}_pocket{index}.mmcif"), PocketSelect(pocket))

def produce_cleaned_dataset(structure_dict, pair_path, dist, target):
    """
    Generate cleaned dataset in out_path, given dictionary of structures processed by process_files.
    :param structure_dict: (dict) dictionary containing each structure, keyed by PDB code. Each PDB is a dict containing protein in Biopython format and ligand in RDKit Mol format
    :param pair_path: (string) path to directory for protein, target, ligand group
    :param dist: (float) distance cutoff for defining pocket
    :param target: (string) target ligand name
    :return:
    """
    protein = structure_dict['protein']
    for lig_name in structure_dict:
        if lig_name == 'protein':
            continue
        ligand = structure_dict[lig_name]
        # check for failed ligand (due to bad structure file)
        if ligand is None:
            continue
        pocket_res = get_pocket_res(protein, ligand, dist)
        write_files(protein, pocket_res, pair_path, target, lig_name[3:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either all, group, check, or MAPK14')
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('--index', type=int, default=-1, help='for group task, group number')
    parser.add_argument('--dist', type=float, default=6.0, help='distance cutoff for defining pocket')
    args = parser.parse_args()
    
    if not os.path.exists(args.raw_root):
        raise Exception('Path not found. Please enter valid path to PDBBind dataset.')

    if args.task == 'all':
        process = get_prots(args.docked_prot_file)
        grouped_files = group_files(N, process)

        if not os.path.exists(args.run_path):
            os.mkdir(args.run_path)

        for i, group in enumerate(grouped_files):
            cmd = 'sbatch -p owners -t 1:00:00 -o {} --wrap="$SCHRODINGER/run python3 process_pdbbind.py group {} {} {} ' \
                  '--index {}"'
            os.system(cmd.format(os.path.join(args.run_path, 'process{}.out'.format(i)), args.docked_prot_file,
                                 args.run_path, args.raw_root, i))

    if args.task == 'group':
        process = get_prots(args.docked_prot_file)
        grouped_files = group_files(N, process)

        for protein, target, start in grouped_files[args.index]:
            protein_path = os.path.join(args.raw_root, protein)
            pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
            structures = process_files(pair_path)
            produce_cleaned_dataset(structures, pair_path, args.dist, target)

    if args.task == 'check':
        process = []
        num_pairs = 0
        with open(args.docked_prot_file) as fp:
            for line in tqdm(fp, desc='going through protein, target, start groups'):
                if line[0] == '#': continue
                protein, target, start = line.strip().split()
                num_pairs += 1
                protein_path = os.path.join(args.raw_root, protein)
                pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
                pocket_path = os.path.join(pair_path, 'pockets')

                pv_file = os.path.join(pair_path, '{}-to-{}_pv.maegz'.format(target, start))
                num_poses = min(MAX_POSES, len(list(structure.StructureReader(pv_file))))
                # num_poses = 0
                for i in range(MAX_DECOYS):
                    if not os.path.join(pocket_path, '{}_pocket{}.mmcif'.format(target, str(num_poses) +
                                                                                        chr(ord('a') + i))):
                        process.append((protein, target, start))
                        break

        print('Missing', len(process), '/', num_pairs)
        print(process)

    if args.task == 'remove_pv':
        with open(args.docked_prot_file) as fp:
            for line in tqdm(fp, desc='going through protein, target, start groups'):
                if line[0] == '#': continue
                protein, target, start = line.strip().split()
                protein_path = os.path.join(args.raw_root, protein)
                pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
                os.remove(os.path.join(pair_path, '{}-to-{}_glide_pv.maegz'.format(target, start)))

    if args.task == 'remove_pockets':
        process = get_prots(args.docked_prot_file)
        grouped_files = group_files(N, process)
        for i, group in enumerate(grouped_files):
            with open(os.path.join(args.run_path, 'remove_pocket{}_in.sh'.format(i)), 'w') as f:
                f.write('#!/bin/bash\n')
                for protein, target, start in group:
                    protein_path = os.path.join(args.raw_root, protein)
                    pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
                    pocket_path = os.path.join(pair_path, 'pockets')

                    if os.path.exists(pocket_path):
                        f.write('rm -r {}\n'.format(pocket_path))
            os.chdir(args.run_path)
            os.system('sbatch -p owners -t 00:30:00 -o remove_pocket{}.out remove_pocket{}_in.sh'.format(i, i))

    if args.task == 'check_remove_pockets':
        process = []
        num_pairs = 0
        with open(args.docked_prot_file) as fp:
            for line in tqdm(fp, desc='going through protein, target, start groups'):
                if line[0] == '#': continue
                protein, target, start = line.strip().split()
                num_pairs += 1
                protein_path = os.path.join(args.raw_root, protein)
                pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
                pocket_path = os.path.join(pair_path, 'pockets')

                if os.path.exists(pocket_path):
                    process.append((protein, target, start))

        print('Missing', len(process), '/', num_pairs)
        print(process)

if __name__ == "__main__":
    main()

