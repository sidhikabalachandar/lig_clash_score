"""
The purpose of this code is to create the pytorch-geometric graphs, create the Data files, and to load the
train/val/test data

It can be run on sherlock using
$ /home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python graph.py all /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/models/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/combined.csv --no_protein

/home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python graph.py group /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits/combined_index_balance_clash_large.txt /home/users/sidhikab/lig_clash_score/src/models/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/combined_conformer_poses.csv --decoy_type conformer_poses --score_feature --index 0

$ /home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python graph.py group /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/models/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/combined.csv --index 0 --no_protein
$ /home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python graph.py check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/models/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/combined.csv --no_protein
$ /home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python graph.py update /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/models/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/combined.csv --new_prot_file /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random2.txt
$ /home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python graph.py remove /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/models/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/combined.csv
"""

import sys
sys.path[-2] = '/home/users/sidhikab/lig_clash_score/src'
from util import datatypes as dt
from util import graph

import pandas as pd
import os
from tqdm import tqdm
import argparse
import pickle
import random
import scipy.spatial
from rdkit import Chem
import Bio.PDB
from Bio.PDB.PDBIO import Select

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

def get_mcss(pdb, label_df):
    """
    searches for pdb's rmsd in combined rmsd df
    :param pdb: (string) {target}_lig{id}
    :param label_df: (df) combined rmsd df
    :return: (float) rmsd value
    """
    return label_df[label_df['target'] == pdb]['mcss'].iloc[0]

def get_score_no_vdw(pdb, label_df):
    """
    searches for pdb's rmsd in combined rmsd df
    :param pdb: (string) {target}_lig{id}
    :param label_df: (df) combined rmsd df
    :return: (float) rmsd value
    """
    return label_df[label_df['target'] == pdb]['target_start_score_no_vdw'].iloc[0]

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

def create_graphs(target, start, pair_path, decoy_type, dist, label_file, include_score, lower_score_bound, upper_score_bound,
                  include_protein):
    """
    Creates pytorch-geometric graph objects for target, start pair
    :param target: (string) name of target ligand
    :param start: (string) name of start ligand
    :param pair_path: (string) path to directory with target/start info
    :return:
    """
    data = {}
    error_count = 0
    io = Bio.PDB.MMCIFIO()
    pose_path = os.path.join(pair_path, decoy_type)
    receptor_file = os.path.join(pair_path, '{}_prot.pdb'.format(start))

    label_df = pd.read_csv(label_file)
    protein = dt.read_any(receptor_file)
    io.set_structure(protein)

    for file in tqdm(os.listdir(pose_path), desc='files'):
        if file[-3:] == 'sdf':
            pdb_code = file[:-4]
            index = pdb_code.split('_lig')[-1]
            ligand = get_ligand(os.path.join(pose_path, file))
            pocket = get_pocket_res(protein, ligand, dist)

            if len(pocket) != 0:
                if include_protein:
                    io.save(os.path.join(pair_path, f"{target}_pocket{index}.mmcif"), PocketSelect(pocket))
                    score = get_score_no_vdw(pdb_code, label_df)
                    if score < lower_score_bound:
                        score = lower_score_bound
                    if score > upper_score_bound:
                        score = upper_score_bound
                    prot_graph = graph.prot_df_to_graph(dt.bp_to_df(dt.read_any(
                        os.path.join(pair_path, f"{target}_pocket{index}.mmcif"), name=pdb_code)),
                        get_mcss(pdb_code, label_df), score, include_score=include_score)
                    mol_graph = graph.mol_to_graph(dt.read_sdf_to_mol(os.path.join(pose_path, file))[0])
                    node_feats, edge_index, edge_feats, pos = graph.combine_graphs(prot_graph, mol_graph, edges_between=True)
                    data[pdb_code] = (node_feats, edge_index, edge_feats, pos)
                    os.remove(os.path.join(pair_path, f"{target}_pocket{index}.mmcif"))
                else:
                    mol_graph = graph.mol_to_graph(dt.read_sdf_to_mol(os.path.join(pose_path, file))[0])
                    node_feats, edge_index, edge_feats, pos = mol_graph
                    data[pdb_code] = (node_feats, edge_index, edge_feats, pos)

            else:
                error_count += 1

    print(len(data))
    print(error_count)
    if include_score:
        outfile = open(os.path.join(pair_path, '{}-to-{}_{}_graph_with_score.pkl'.format(target, start, decoy_type)), 'wb')
    elif not include_protein:
        outfile = open(os.path.join(pair_path, '{}-to-{}_{}_graph_without_protein.pkl'.format(target, start, decoy_type)), 'wb')
    else:
        outfile = open(os.path.join(pair_path, '{}-to-{}_{}_graph.pkl'.format(target, start, decoy_type)), 'wb')

    pickle.dump(data, outfile)

def run_all(docked_prot_file, run_path, root, label_file, decoy_type, grouped_files, n, include_score, include_protein):
    for i, group in enumerate(grouped_files):
        cmd = 'sbatch -p owners -t 1:00:00 -o {} --wrap="' \
              '/home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python graph.py group ' \
              '{} {} {} {} --n {} --index {} --decoy_type {}'
        if include_score:
            cmd += ' --score_feature'
        if not include_protein:
            cmd += ' --no_protein'

        cmd += '"'
        os.system(cmd.format(os.path.join(run_path, 'graph{}.out'.format(i)), docked_prot_file, run_path, root,
                                 label_file, n, i, decoy_type))

def run_group(grouped_files, raw_root, label_file, decoy_type, index, dist, include_score, lower_score_bound, upper_score_bound,
              include_protein):
    for protein, target, start in grouped_files[index]:
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
        print(pair_path)
        create_graphs(target, start, pair_path, decoy_type, dist, label_file, include_score, lower_score_bound,
                      upper_score_bound, include_protein)

def run_check(raw_root, docked_prot_file, decoy_type, include_score, include_protein):
    process = []
    num_pairs = 0
    with open(docked_prot_file) as fp:
        for line in tqdm(fp, desc='going through protein, target, start groups'):
            if line[0] == '#': continue
            protein, target, start = line.strip().split()
            pair = '{}-to-{}'.format(target, start)
            num_pairs += 1
            protein_path = os.path.join(raw_root, protein)
            pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
            if include_score and not os.path.exists(
                    os.path.join(pair_path, '{}_{}_graph_with_score.pkl'.format(pair, decoy_type))):
                process.append((protein, target, start))
            elif not include_protein and not os.path.exists(
                    os.path.join(pair_path, '{}_{}_graph_without_protein.pkl'.format(pair, decoy_type))):
                process.append((protein, target, start))
            elif not os.path.exists(os.path.join(pair_path, '{}_{}_graph.pkl'.format(pair, decoy_type))):
                process.append((protein, target, start))

    print('Missing', len(process), '/', num_pairs)
    print(process)

def update(docked_prot_file, raw_root, new_prot_file):
    """
    update index by removing protein, target, start that could not create grids
    :param docked_prot_file: (string) file listing proteins to process
    :param raw_root: (string) directory where raw data will be placed
    :param new_prot_file: (string) name of new prot file
    :return:
    """
    text = []
    with open(docked_prot_file) as fp:
        for line in tqdm(fp, desc='files'):
            if line[0] == '#': continue
            protein, target, start = line.strip().split()
            protein_path = os.path.join(raw_root, protein)
            pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
            if os.path.exists('{}/{}-to-{}_graph.pkl'.format(pair_path, target, start)):
                text.append(line)

    file = open(new_prot_file, "w")
    file.writelines(text)
    file.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either all, group, check')
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('root', type=str, help='directory where raw and processed directories can be found')
    parser.add_argument('label_file', type=str, help='file with rmsd labels')
    parser.add_argument('--index', type=int, default=-1, help='for group task, group number')
    parser.add_argument('--index_file', type=str, default=os.path.join(os.getcwd(), 'index_groups.pkl'),
                        help='for combine task, file with protein, target, start, and starting index information for '
                             'each group')
    parser.add_argument('--decoy_type', type=str, default='ligand_poses', help='either cartesian_poses, ligand_poses, '
                                                                               'or conformer_poses')
    parser.add_argument('--dist', type=float, default=6.0, help='distance cutoff for defining pocket')
    parser.add_argument('--cutoff', type=float, default=0.1, help='proportion of pdbbind data used')
    parser.add_argument('--lower_score_bound', type=float, default=-20, help='any physics score below this value, will '
                                                                             'be set to this value')
    parser.add_argument('--upper_score_bound', type=float, default=20, help='any physics score above this value, will '
                                                                            'be set to this value')
    parser.add_argument('--n', type=int, default=3, help='number of protein, target, start groups processed in '
                                                         'group task')
    parser.add_argument('--new_prot_file', type=str, default=os.path.join(os.getcwd(), 'index.txt'),
                        help='for update task, name of new prot file')
    parser.add_argument('--score_feature', dest='include_score', action='store_true')
    parser.add_argument('--no_score_feature', dest='include_score', action='store_false')
    parser.set_defaults(include_score=False)
    parser.add_argument('--protein', dest='include_protein', action='store_true')
    parser.add_argument('--no_protein', dest='include_protein', action='store_false')
    parser.set_defaults(include_protein=True)
    args = parser.parse_args()

    raw_root = os.path.join(args.root, 'raw')
    if not args.include_score:
        processed_root = os.path.join(args.root, 'processed')
    else:
        processed_root = os.path.join(args.root, 'processed_score', 'processed')
        if not os.path.exists(os.path.join(args.root, 'processed_score')):
            os.mkdir(os.path.join(args.root, 'processed_score'))
        if not os.path.exists(os.path.join(args.root, 'processed_score', 'raw')):
            os.mkdir(os.path.join(args.root, 'processed_score', 'raw'))
    random.seed(0)

    if not os.path.exists(processed_root):
        os.mkdir(processed_root)

    if not os.path.exists(args.run_path):
        os.mkdir(args.run_path)

    if args.task == 'all':
        process = get_prots(args.docked_prot_file)
        grouped_files = group_files(args.n, process)
        run_all(args.docked_prot_file, args.run_path, args.root, args.label_file, args.decoy_type, grouped_files,
                args.n, args.include_score, args.include_protein)

    if args.task == 'group':
        process = get_prots(args.docked_prot_file)
        grouped_files = group_files(args.n, process)
        run_group(grouped_files, raw_root, args.label_file, args.decoy_type, args.index, args.dist, args.include_score,
                  args.lower_score_bound, args.upper_score_bound, args.include_protein)

    if args.task == 'check':
        run_check(raw_root, args.docked_prot_file, args.decoy_type, args.include_score, args.include_protein)

    if args.task == 'update':
        update(args.docked_prot_file, raw_root, args.new_prot_file)

    if args.task == 'remove':
        with open(args.docked_prot_file) as fp:
            for line in tqdm(fp, desc='files'):
                if line[0] == '#': continue
                protein, target, start = line.strip().split()
                protein_path = os.path.join(raw_root, protein)
                pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
                file = os.path.join(pair_path, '{}-to-{}_graph.pkl'.format(target, start))
                if os.path.exists(file):
                    os.remove(file)

                file = os.path.join(pair_path, '{}-to-{}_graph_with_score.pkl'.format(target, start))
                if os.path.exists(file):
                    os.remove(file)

if __name__=="__main__":
    main()



