"""
The purpose of this code is to create the pytorch-geometric graphs, create the Data files, and to load the
train/val/test data

It can be run on sherlock using
$ /home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python pdbbind_dataloader.py all /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/models/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/combined.csv
$ /home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python pdbbind_dataloader.py group /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/models/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/combined.csv --index 0
$ /home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python pdbbind_dataloader.py check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/models/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/combined.csv
$ /home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python pdbbind_dataloader.py update /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/models/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/combined.csv --new_prot_file /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random2.txt
$ /home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python pdbbind_dataloader.py combine_all /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/models/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/combined.csv
$ /home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python pdbbind_dataloader.py combine_group /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/models/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/combined.csv --index 0
$ /home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python pdbbind_dataloader.py combine_check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/models/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/combined.csv
$ /home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python pdbbind_dataloader.py pdbbind_dataloader /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/models/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/combined.csv
"""

import sys
sys.path[-2] = '/home/users/sidhikab/lig_clash_score/src'
from util import datatypes as dt
from util import splits as sp
from util import graph

import pandas as pd
import os
import torch
from torch_geometric.data import Dataset, Data, DataLoader
from tqdm import tqdm
import argparse
import pickle
import random
import scipy.spatial
from rdkit import Chem
import Bio.PDB
from Bio.PDB.PDBIO import Select

# loader for pytorch-geometric
class GraphPDBBind(Dataset):
    """
    PDBBind dataset in pytorch-geometric format. 
    Ref: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/data/dataset.html#Dataset
    """
    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphPDBBind, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        f = open("/home/users/sidhikab/lig_clash_score/src/models/out/test.out", "a")
        f.write('getting raw file names\n')
        f.close()
        return sorted(os.listdir(self.raw_dir))

    @property
    def processed_file_names(self):
        f = open("/home/users/sidhikab/lig_clash_score/src/models/out/test.out", "a")
        f.write('getting processed file names\n')
        f.close()
        return sorted(os.listdir(self.processed_dir))

    def process(self):
        f = open("/home/users/sidhikab/lig_clash_score/src/models/out/test.out", "a")
        f.write('processing\n')
        f.close()
        pass

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data

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

def get_index_groups(process, pkl_file, label_file, raw_root, cutoff):
    """
    gets list of all protein, target ligands, starting ligands, and starting indices information in the index file (up to
    CUTOFF)
    :param process: (list) shuffled list of all protein, target ligands, and starting ligands to process
    :param pkl_file: (string) file containing list of all protein, target ligands, starting ligands, and starting
                                indices information (or file path where this information will be saved)
    :param label_file: (string) file containing rmsd label information
    :param raw_root: (string) path to directory with data
    :return: grouped_files (list) list of all protein, target ligands, starting ligands, and starting indices to process
    """
    if (os.path.exists(pkl_file)):
        infile = open(pkl_file, 'rb')
        index_groups = pickle.load(infile)
        infile.close()
        return index_groups
    else:
        label_df = pd.read_csv(label_file)
        index_groups = []
        num_pairs = 0
        num_codes = 0

        for protein, target, start in tqdm(process, desc='going through protein, target, start groups'):
            if num_codes / len(label_df) > cutoff:
                break
            num_pairs += 1
            index_groups.append((protein, target, start, num_codes))

            #update num_codes
            protein_path = os.path.join(raw_root, protein)
            pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
            graph_dir = '{}/{}-to-{}_graph.pkl'.format(pair_path, target, start)
            infile = open(graph_dir, 'rb')
            graph_data = pickle.load(infile)
            infile.close()
            for _ in graph_data:
                num_codes += 1

        outfile = open(pkl_file, 'wb')
        pickle.dump(index_groups, outfile)
        print(num_codes)
        print(num_pairs)

        return index_groups

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

def get_label(pdb, label_df):
    """
    searches for pdb's rmsd in combined rmsd df
    :param pdb: (string) {target}_lig{id}
    :param label_df: (df) combined rmsd df
    :return: (float) rmsd value
    """
    return label_df[label_df['target'] == pdb]['rmsd'].iloc[0]

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
    return label_df[label_df['target'] == pdb]['score_no_vdw'].iloc[0]

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

def create_graphs(target, start, pair_path, dist, label_file):
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
    pose_path = os.path.join(pair_path, 'ligand_poses')
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
                io.save(os.path.join(pair_path, f"{target}_pocket{index}.mmcif"), PocketSelect(pocket))
                prot_graph = graph.prot_df_to_graph(dt.bp_to_df(dt.read_any(
                    os.path.join(pair_path, f"{target}_pocket{index}.mmcif"), name=pdb_code)),
                    get_mcss(pdb_code, label_df))
                mol_graph = graph.mol_to_graph(dt.read_sdf_to_mol(os.path.join(pose_path, file))[0])
                node_feats, edge_index, edge_feats, pos = graph.combine_graphs(prot_graph, mol_graph, edges_between=True)
                data[pdb_code] = (node_feats, edge_index, edge_feats, pos)
                os.remove(os.path.join(pair_path, f"{target}_pocket{index}.mmcif"))
            else:
                error_count += 1

    print(len(data))
    print(error_count)
    outfile = open(os.path.join(pair_path, '{}-to-{}_graph.pkl'.format(target, start)), 'wb')
    pickle.dump(data, outfile)

def split_process(protein, target, start, label_file, pair_path, processed_root, start_index):
    """
    creates Data file for target/start pair
    :param target: (string) name of target ligand
    :param start: (string) name of start ligand
    :param label_file: (string) file containing rmsd label information
    :param pair_path: (string) path to directory with target/start info
    :param processed_root: (string) directory where data files will be written to
    :param start_index: (int) starting index for labeling data files for target/start pair
    :return: grouped_files (list) list of sublists of pairs
    """
    label_df = pd.read_csv(label_file)
    graph_dir = '{}/{}-to-{}_graph.pkl'.format(pair_path, target, start)
    infile = open(graph_dir, 'rb')
    graph_data = pickle.load(infile)
    infile.close()
    print(len(graph_data))
    for pdb_code in tqdm(graph_data, desc='pdb_codes'):
        node_feats, edge_index, edge_feats, pos = graph_data[pdb_code]
        y = torch.FloatTensor([get_label(pdb_code, label_df)])
        data = Data(node_feats, edge_index, edge_feats, y=y, pos=pos)
        data.pdb = '{}_{}-to-{}_{}'.format(protein, target, start, pdb_code)
        data.physics_score = get_score_no_vdw(pdb_code, label_df)
        torch.save(data, os.path.join(processed_root, 'data_{}.pt'.format(start_index)))
        start_index += 1

def pdbbind_dataloader(batch_size, output_file, data_dir='../../data/pdbbind', split_file=None):
    """
    Creates dataloader for PDBBind dataset with specified split.
    Assumes pre-computed split in 'split_file', which is used to index Dataset object
    :param batch_size: (int) size of each batch of data
    :param data_dir: (string) root directory of GraphPDBBind class
    :param split_file: (string) file with pre-computed split information
    :return: (dataloader) dataloader for PDBBind dataset with specified split
    """
    f = open(output_file, "a")
    f.write('initializing graph\n')
    f.close()
    dataset = GraphPDBBind(root=data_dir)
    if split_file is None:
        return DataLoader(dataset, batch_size, shuffle=True)
    f = open(output_file, "a")
    f.write('reading split file\n')
    f.close()
    indices = sp.read_split_file(split_file)
    f = open(output_file, "a")
    f.write('creating data loader\n')
    f.close()
    dl = DataLoader(dataset.index_select(indices), batch_size, shuffle=True)
    f = open(output_file, "a")
    f.write('finished creating data loader\n')
    f.close()
    return dl

def run_all(docked_prot_file, run_path, root, label_file, grouped_files, n):
    for i, group in enumerate(grouped_files):
        cmd = 'sbatch -p owners -t 1:00:00 -o {} --wrap="' \
              '/home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python pdbbind_dataloader.py group ' \
              '{} {} {} {} --n {} --index {}"'
        os.system(cmd.format(os.path.join(run_path, 'graph{}.out'.format(i)), docked_prot_file, run_path, root,
                             label_file, n, i))

def run_group(grouped_files, raw_root, label_file, index, dist):
    for protein, target, start in grouped_files[index]:
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
        create_graphs(target, start, pair_path, dist, label_file)

def run_check(raw_root, docked_prot_file):
    process = []
    num_pairs = 0
    with open(docked_prot_file) as fp:
        for line in tqdm(fp, desc='going through protein, target, start groups'):
            if line[0] == '#': continue
            protein, target, start = line.strip().split()
            num_pairs += 1
            protein_path = os.path.join(raw_root, protein)
            pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
            if not os.path.exists(os.path.join(pair_path, '{}-to-{}_graph.pkl'.format(target, start))):
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

def run_combine_all(docked_prot_file, run_path, root, label_file, grouped_files, n):
    for i, group in enumerate(grouped_files):
        cmd = 'sbatch -p owners -t 1:00:00 -o {} --wrap="' \
              '/home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python pdbbind_dataloader.py ' \
              'combine_group {} {} {} {} --n {} --index {}"'
        os.system(cmd.format(os.path.join(run_path, 'processed{}.out'.format(i)), docked_prot_file, run_path, root,
                             label_file, n, i))

def run_combine_group(grouped_files, raw_root, processed_root, label_file, index):
    for protein, target, start, start_index in grouped_files[index]:
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
        split_process(protein, target, start, label_file, pair_path, processed_root, start_index)

def run_combine_check(raw_root, processed_root, docked_prot_file, label_file, cutoff):
    process = get_prots(docked_prot_file)
    random.shuffle(process)

    label_df = pd.read_csv(label_file)
    num_codes = 0
    num_pairs = 0
    index_groups = []

    for protein, target, start in tqdm(process, desc='going through protein, target, start groups'):
        if num_codes / len(label_df) > cutoff:
            break

        num_pairs += 1

        # update num_codes
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
        graph_dir = '{}/{}-to-{}_graph.pkl'.format(pair_path, target, start)
        infile = open(graph_dir, 'rb')
        graph_data = pickle.load(infile)
        infile.close()
        for _ in graph_data:
            if not os.path.exists(os.path.join(processed_root, 'data_{}.pt'.format(num_codes))):
                index_groups.append((protein, target, start, num_codes))
                break
            num_codes += 1

    print('Missing', len(index_groups), '/', num_pairs)
    print(index_groups)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either all, group, check, MAPK14, combine_all, combine_group, '
                                               'combine_check, or pdbbind_dataloader')
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('root', type=str, help='directory where raw and processed directories can be found')
    parser.add_argument('label_file', type=str, help='file with rmsd labels')
    parser.add_argument('--index', type=int, default=-1, help='for group task, group number')
    parser.add_argument('--index_file', type=str, default=os.path.join(os.getcwd(), 'index_groups.pkl'),
                        help='for combine task, file with protein, target, start, and starting index information for '
                             'each group')
    parser.add_argument('--dist', type=float, default=6.0, help='distance cutoff for defining pocket')
    parser.add_argument('--cutoff', type=float, default=0.1, help='proportion of ')
    parser.add_argument('--n', type=int, default=3, help='number of protein, target, start groups processed in '
                                                         'group task')
    parser.add_argument('--new_prot_file', type=str, default=os.path.join(os.getcwd(), 'index.txt'),
                        help='for update task, name of new prot file')
    args = parser.parse_args()

    raw_root = os.path.join(args.root, 'raw')
    processed_root = os.path.join(args.root, 'processed')
    random.seed(0)

    if not os.path.exists(processed_root):
        os.mkdir(processed_root)

    if not os.path.exists(args.run_path):
        os.mkdir(args.run_path)

    if args.task == 'all':
        process = get_prots(args.docked_prot_file)
        grouped_files = group_files(args.n, process)
        run_all(args.docked_prot_file, args.run_path, args.root, args.label_file, grouped_files, args.n)

    if args.task == 'group':
        process = get_prots(args.docked_prot_file)
        grouped_files = group_files(args.n, process)
        run_group(grouped_files, raw_root, args.label_file, args.index, args.dist)

    if args.task == 'check':
        run_check(raw_root, args.docked_prot_file)

    if args.task == 'update':
        update(args.docked_prot_file, raw_root, args.new_prot_file)

    if args.task == 'combine_all':
        process = get_prots(args.docked_prot_file)
        random.shuffle(process)
        index_groups = get_index_groups(process, args.index_file, args.label_file, raw_root, args.cutoff)
        grouped_files = group_files(args.n, index_groups)
        run_combine_all(args.docked_prot_file, args.run_path, args.root, args.label_file, grouped_files, args.n)

    if args.task == 'combine_group':
        process = get_prots(args.docked_prot_file)
        random.shuffle(process)
        index_groups = get_index_groups(process, args.index_file, args.label_file, raw_root, args.cutoff)
        grouped_files = group_files(args.n, index_groups)
        run_combine_group(grouped_files, raw_root, processed_root, args.label_file, args.index)

    if args.task == 'combine_check':
        run_combine_check(raw_root, processed_root, args.docked_prot_file, args.label_file, args.cutoff)

    if args.task == 'pdbbind_dataloader':
        loader = pdbbind_dataloader(1, "/home/users/sidhikab/lig_clash_score/src/models/out/test.out", data_dir=args.root)
        print(len(loader))
        for data in loader:
            print(data)
            print(data.pdb)
            break

if __name__=="__main__":
    main()



