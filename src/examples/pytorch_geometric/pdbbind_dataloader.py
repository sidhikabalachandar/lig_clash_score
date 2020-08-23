"""
The purpose of this code is to create the pytorch-geometric graphs, create the Data files, and to load the
train/val/test data

It can be run on sherlock using
$ /home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python pdbbind_dataloader.py all /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/examples/pytorch_geometric/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/pdbbind_refined_set_labels.csv
$ /home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python pdbbind_dataloader.py group /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/examples/pytorch_geometric/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/pdbbind_refined_set_labels.csv --index 0
$ /home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python pdbbind_dataloader.py check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/examples/pytorch_geometric/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/pdbbind_refined_set_labels.csv
$ /home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python pdbbind_dataloader.py MAPK14
$ /home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python pdbbind_dataloader.py combine_all /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/examples/pytorch_geometric/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/pdbbind_refined_set_labels.csv
$ /home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python pdbbind_dataloader.py combine_group /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/examples/pytorch_geometric/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/pdbbind_refined_set_labels.csv --index 0
$ /home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python pdbbind_dataloader.py combine_check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/examples/pytorch_geometric/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/pdbbind_refined_set_labels.csv
$ /home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python pdbbind_dataloader.py pdbbind_dataloader /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/examples/pytorch_geometric/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/pdbbind_refined_set_labels.csv
"""

import sys
sys.path[-2] = '/home/users/sidhikab/lig_clash_score/src'
from atom3d.util import datatypes as dt
from atom3d.util import splits as sp
from atom3d.protein_ligand.get_labels import get_label
from atom3d.util import graph

import pandas as pd
import os
import torch
from torch_geometric.data import Dataset, Data, DataLoader
from tqdm import tqdm
import argparse
import pickle
import random

CUTOFF = 0.1
N = 3

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
        return sorted(os.listdir(self.raw_dir))

    @property
    def processed_file_names(self):
        return sorted(os.listdir(self.processed_dir))

    def process(self):
        pass

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data


def pdbbind_dataloader(batch_size, data_dir='../../data/pdbbind', split_file=None):
    """
    Creates dataloader for PDBBind dataset with specified split. 
    Assumes pre-computed split in 'split_file', which is used to index Dataset object
    :param batch_size: (int) size of each batch of data
    :param data_dir: (string) root directory of GraphPDBBind class
    :param split_file: (string) file with pre-computed split information
    :return: (dataloader) dataloader for PDBBind dataset with specified split
    """
    dataset = GraphPDBBind(root=data_dir)
    if split_file is None:
        return DataLoader(dataset, batch_size, shuffle=True)
    indices = sp.read_split_file(split_file)
    return DataLoader(dataset.index_select(indices), batch_size, shuffle=True)

def create_graphs(target, start, pair_path):
    """
    Creates pytorch-geometric graph objects for target, start pair
    :param target: (string) name of target ligand
    :param start: (string) name of start ligand
    :param pair_path: (string) path to directory with target/start info
    :return:
    """
    error_count = 0
    data = {}
    with open('{}/{}-to-{}_mcss.csv'.format(pair_path, target, start)) as f:
        mcss = int(f.readline().strip().split(',')[4])
    pose_path = os.path.join(pair_path, 'ligand_poses')
    pocket_dir = os.path.join(pair_path, 'pockets')
    for file in tqdm(os.listdir(pocket_dir), desc='files'):
        try:
            id_code = file.split('_pocket')[-1].split('.mmcif')[0]
            pdb_code = '{}_lig{}'.format(target, id_code)
            prot_graph = graph.prot_df_to_graph(
                dt.bp_to_df(dt.read_any(os.path.join(pocket_dir, file), name=pdb_code)), mcss)
            id_code = file.split('_pocket')[-1].split('.mmcif')[0]
            mol_graph = graph.mol_to_graph(
                dt.read_sdf_to_mol(os.path.join(pose_path, '{}_lig{}.sdf'.format(target, id_code)))[0])
            node_feats, edge_index, edge_feats, pos = graph.combine_graphs(prot_graph, mol_graph,
                                                                           edges_between=True)
            data[pdb_code] = (node_feats, edge_index, edge_feats, pos)

        except Exception as e:
            error_count += 1

    print(len(data))
    outfile = open(os.path.join(pair_path, '{}-to-{}_graph.pkl'.format(target, start)), 'wb')
    pickle.dump(data, outfile)
    print(error_count)

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
        torch.save(data, os.path.join(processed_root, 'data_{}.pt'.format(start_index)))
        start_index += 1

def get_index_groups(process, pkl_file, label_file, raw_root):
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
            if num_codes / len(label_df) > CUTOFF:
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
    args = parser.parse_args()

    raw_root = os.path.join(args.root, 'raw')
    processed_root = os.path.join(args.root, 'processed')
    random.seed(0)

    if not os.path.exists(processed_root):
        os.mkdir(processed_root)

    if args.task == 'all':
        process = get_prots(args.docked_prot_file)
        grouped_files = group_files(N, process)

        if not os.path.exists(args.run_path):
            os.mkdir(args.run_path)

        for i, group in enumerate(grouped_files):
            cmd = 'sbatch -p owners -t 1:00:00 -o {} --wrap="' \
                  '/home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python pdbbind_dataloader.py group ' \
                  '{} {} {} {} --index {}"'
            os.system(cmd.format(os.path.join(args.run_path, 'graph{}.out'.format(i)), args.docked_prot_file,
                                 args.run_path, args.root, args.label_file, i))

    if args.task == 'group':
        process = get_prots(args.docked_prot_file)
        grouped_files = group_files(N, process)

        for protein, target, start in grouped_files[args.index]:
            protein_path = os.path.join(raw_root, protein)
            pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
            create_graphs(target, start, pair_path)

    if args.task == 'check':
        process = []
        num_pairs = 0
        with open(args.docked_prot_file) as fp:
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

    if args.task == 'MAPK14':
        ligs = ['3D83', '4F9Y']
        for target in ligs:
            for start in ligs:
                if target != start:
                    protein_path = os.path.join(raw_root, protein)
                    pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
                    create_graphs(target, start, pair_path)

    if args.task == 'combine_all':
        process = get_prots(args.docked_prot_file)
        random.shuffle(process)
        index_groups = get_index_groups(process, args.index_file, args.label_file, raw_root)
        grouped_files = group_files(N, index_groups)

        if not os.path.exists(args.run_path):
            os.mkdir(args.run_path)

        for i, group in enumerate(grouped_files):
            cmd = 'sbatch -p owners -t 1:00:00 -o {} --wrap="' \
                  '/home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python pdbbind_dataloader.py ' \
                  'combine_group {} {} {} {} --index {}"'
            os.system(cmd.format(os.path.join(args.run_path, 'processed{}.out'.format(i)), args.docked_prot_file,
                                 args.run_path, args.root, args.label_file, i))

    if args.task == 'combine_group':
        process = get_prots(args.docked_prot_file)
        random.shuffle(process)
        index_groups = get_index_groups(process, args.index_file, args.label_file, raw_root)
        grouped_files = group_files(N, index_groups)

        for protein, target, start, start_index in grouped_files[args.index]:
            protein_path = os.path.join(raw_root, protein)
            pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
            print(pair_path)
            split_process(protein, target, start, args.label_file, pair_path, processed_root, start_index)

    if args.task == 'combine_check':
        process = get_prots(args.docked_prot_file)
        random.shuffle(process)

        label_df = pd.read_csv(args.label_file)
        num_codes = 0
        num_pairs = 0
        index_groups = []

        for protein, target, start in tqdm(process, desc='going through protein, target, start groups'):
            if num_codes / len(label_df) > CUTOFF:
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

    if args.task == 'pdbbind_dataloader':
        data = pdbbind_dataloader(1, data_dir=args.root)
        print(len(data))

if __name__=="__main__":
    main()



