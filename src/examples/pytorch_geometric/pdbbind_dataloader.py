"""
The purpose of this code is to train the gnn model
It can be run on sherlock using
$ /home/groups/rondror/software/sidhikab/miniconda/envs/geometric/bin/python pdbbind_dataloader.py all /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/graphs /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt
$ /home/groups/rondror/software/sidhikab/miniconda/envs/geometric/bin/python pdbbind_dataloader.py group /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/graphs /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt --group 0
$ /home/groups/rondror/software/sidhikab/miniconda/envs/geometric/bin/python pdbbind_dataloader.py check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/graphs /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt
$ /home/groups/rondror/software/sidhikab/miniconda/envs/geometric/bin/python pdbbind_dataloader.py combine /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/graphs /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt
"""

# import numpy as np
import pandas as pd
import sys
sys.path.append('../..')
from atom3d.util import datatypes as dt
from atom3d.util import file as fi
from atom3d.util import splits as sp
from atom3d.protein_ligand.get_labels import get_label
from atom3d.util import graph
import os
import torch
from torch_geometric.data import Dataset, Data, DataLoader
from tqdm import tqdm
import argparse

# import logging
# import pdb
import pickle

graph_root = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/graphs'
protein_file = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt'
run_path = '/home/users/sidhikab/flexibility_project/atom3d/src/atom3d/protein_ligand/run'

# loader for pytorch-geometric

class GraphPDBBind(Dataset):
    """
    PDBBind dataset in pytorch-geometric format. 
    Ref: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/data/dataset.html#Dataset
    """
    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphPDBBind, self).__init__(root, transform, pre_transform)

        self.pdb_idx_dict = self.get_idx_mapping()
        self.idx_pdb_dict = {v:k for k,v in self.pdb_idx_dict.items()}

    @property
    def raw_file_names(self):
        return sorted(os.listdir(self.raw_dir))

    @property
    def processed_file_names(self):
        num_samples = len(self.raw_file_names) // 3 # each example has protein/pocket/ligand files
        return [f'data_{i}.pt' for i in range(num_samples)]

    def get_idx_mapping(self):
        pdb_idx_dict = {}
        i = 0
        for file in self.raw_file_names:
            if '_pocket' in file:
                pdb_code = fi.get_pdb_code(file)
                pdb_idx_dict[pdb_code] = i
                i += 1
        return pdb_idx_dict


    def pdb_to_idx(self, pdb):
        return self.pdb_idx_dict.get(pdb)

    def process(self):
        label_file = os.path.join(self.root, 'pdbbind_refined_set_labels.csv')
        label_df = pd.read_csv(label_file)
        i = 0
        with open(protein_file) as fp:
            for line in tqdm(fp, desc='files'):
                if line[0] == '#': continue
                protein, target, start = line.strip().split()
                graph_dir = '{}/{}-to-{}_graph.pkl'.format(graph_root, target, start)
                infile = open(graph_dir, 'rb')
                data = pickle.load(infile)
                infile.close()
                print(data)
                for pdb_code in data:
                    node_feats, edge_index, edge_feats, pos = data[pdb_code]
                    y = torch.FloatTensor([get_label(pdb_code, label_df)])
                    data = Data(node_feats, edge_index, edge_feats, y=y, pos=pos)
                    data.pdb = pdb_code
                    torch.save(data, os.path.join(self.processed_dir, 'data_{}.pt'.format(i)))
                    print(os.path.join(self.processed_dir, 'data_{}.pt'.format(i)))
                    i += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data


def pdbbind_dataloader(batch_size, data_dir='../../data/pdbbind', split_file=None):
    """
    Creates dataloader for PDBBind dataset with specified split. 
    Assumes pre-computed split in 'split_file', which is used to index Dataset object
    TODO: implement on-the-fly splitting using split functions
    """
    dataset = GraphPDBBind(root=data_dir)
    print(len(dataset))
    if split_file is None:
        return DataLoader(dataset, batch_size, shuffle=True)
    indices = sp.read_split_file(split_file)
    print(indices)

    # if split specifies pdb ids, convert to indices
    if isinstance(indices[0], str):
        indices = [dataset.pdb_to_idx(x) for x in indices if dataset.pdb_to_idx(x)]
        pdb_codes = [x for x in indices if dataset.pdb_to_idx(x)]
    return DataLoader(dataset.index_select(indices), batch_size, shuffle=True)

def create_graphs(target, start, root, out_dir):
    error_count = 0
    data = {}
    target_dir = os.path.join(root, 'processed/{}'.format(target))
    with open('{}/mcss/{}-to-{}_mcss.csv'.format(root, target, start)) as f:
        mcss = int(f.readline().strip().split(',')[4])
    for file in tqdm(os.listdir(target_dir), desc='files'):
        if '_pocket' in file:
            try:
                id_code = file.split('_pocket')[-1].split('.mmcif')[0]
                pdb_code = '{}_lig{}'.format(target, id_code)
                prot_graph = graph.prot_df_to_graph(
                    dt.bp_to_df(dt.read_any(os.path.join(target_dir, file), name=pdb_code)), mcss)
                id_code = file.split('_pocket')[-1].split('.mmcif')[0]
                pdb_code = '{}_lig{}'.format(target, id_code)
                mol_graph = graph.mol_to_graph(
                    dt.read_sdf_to_mol(os.path.join(target_dir, '{}_ligand{}.sdf'.format(target, id_code)),
                                       addHs=True)[0])
                node_feats, edge_index, edge_feats, pos = graph.combine_graphs(prot_graph, mol_graph,
                                                                               edges_between=True)
                data[pdb_code] = (node_feats, edge_index, edge_feats, pos)

            except Exception as e:
                error_count += 1

    outfile = open(os.path.join(out_dir, '{}-to-{}_graph.pkl'.format(target, start)), 'wb')
    pickle.dump(data, outfile)
    print(error_count)

def get_prots(fname, out_dir):
    pairs = []
    unfinished_pairs = []
    with open(fname) as fp:
        for line in tqdm(fp, desc='files'):
            if line[0] == '#': continue
            protein, target, start = line.strip().split()
            pairs.append((protein, target, start))
            graph_dir = '{}/{}-to-{}_graph.pkl'.format(graph_root, target, start)
            infile = open(graph_dir, 'rb')
            data = pickle.load(infile)
            infile.close()
            if len(data) == 0:
                unfinished_pairs.append((protein, target, start))

    return pairs, unfinished_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either all or group')
    parser.add_argument('root', type=str, help='either all or group')
    parser.add_argument('out_dir', type=str, help='either all or group')
    parser.add_argument('prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('--group', type=int, default=-1, help='if type is group, argument indicates group index')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    pairs, unfinished_pairs = get_prots(args.prot_file, args.out_dir)
    n = 3
    grouped_files = []

    for i in range(0, len(pairs), n):
        grouped_files += [pairs[i: i + n]]

    if args.task == 'all':
        for i in range(len(grouped_files)):
            cmd = 'sbatch -p owners -t 5:00:00 -o {} --wrap="' \
                  '/home/groups/rondror/software/sidhikab/miniconda/envs/geometric/bin/python pdbbind_dataloader.py group ' \
                  '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d ' \
                  '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/graphs ' \
                  '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt --group {}"'
            os.system(cmd.format(os.path.join(run_path, 'graph_{}.out'.format(i)), i))
            # print(cmd.format(os.path.join(run_path, 'graph_{}.out'.format(i)), i))
        print(len(grouped_files))

    if args.task == 'group':
        for _, target, start in grouped_files[args.group]:
            create_graphs(target, start, args.root, args.out_dir)

    if args.task == 'check':
        print('Missing:', len(unfinished_pairs), '/', len(pairs))
        # print(unfinished_pairs)

    if args.task == 'combine':
        dataset = GraphPDBBind(root=args.root)

if __name__=="__main__":
    main()



