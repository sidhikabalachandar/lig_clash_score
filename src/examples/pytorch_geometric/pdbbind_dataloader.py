"""
The purpose of this code is to train the gnn model
It can be run on sherlock using
$ /home/groups/rondror/software/sidhikab/miniconda/envs/geometric/bin/python pdbbind_dataloader.py all /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/graphs /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt
$ /home/groups/rondror/software/sidhikab/miniconda/envs/geometric/bin/python pdbbind_dataloader.py group /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/graphs /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt --group 0
$ /home/groups/rondror/software/sidhikab/miniconda/envs/geometric/bin/python pdbbind_dataloader.py pdb /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/graphs /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt --target 4or4 --start 4q46
$ /home/groups/rondror/software/sidhikab/miniconda/envs/geometric/bin/python pdbbind_dataloader.py check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/graphs /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt
$ /home/groups/rondror/software/sidhikab/miniconda/envs/geometric/bin/python pdbbind_dataloader.py combine_all /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/graphs /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt
$ /home/groups/rondror/software/sidhikab/miniconda/envs/geometric/bin/python pdbbind_dataloader.py combine_group /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/graphs /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt
$ /home/groups/rondror/software/sidhikab/miniconda/envs/geometric/bin/python pdbbind_dataloader.py combine_check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/graphs /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt
$ /home/groups/rondror/software/sidhikab/miniconda/envs/geometric/bin/python pdbbind_dataloader.py pdbbind_dataloader /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/graphs /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt
"""

import pandas as pd
import datatypes as dt
import file as fi
import splits as sp
from get_labels import get_label
import graph
import os
import torch
from torch_geometric.data import Dataset, Data, DataLoader
from tqdm import tqdm
import argparse
import pickle

CUTOFF = 0.1
label_file = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/pdbbind_refined_set_labels.csv'
graph_root = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/graphs'
protein_file = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt'
run_path = '/home/users/sidhikab/lig_clash_score/src/examples/pytorch_geometric/run'
processed_root = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/graph_data/processed'

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
        # print(sorted(os.listdir(processed_root)))
        return sorted(os.listdir(processed_root))
        # return []
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
        pass
        # label_df = pd.read_csv(label_file)
        # total_num_codes = len(label_df)
        # num_codes = 0
        # num_pairs = 0
        # with open(protein_file) as fp:
        #     for line in tqdm(fp, desc='files'):
        #         if line[0] == '#': continue
        #         if num_codes / total_num_codes > CUTOFF:
        #             break
        #         protein, target, start = line.strip().split()
        #         num_pairs += 1
        #         graph_dir = '{}/{}-to-{}_graph.pkl'.format(graph_root, target, start)
        #         infile = open(graph_dir, 'rb')
        #         graph_data = pickle.load(infile)
        #         infile.close()
        #         for pdb_code in graph_data:
        #             node_feats, edge_index, edge_feats, pos = graph_data[pdb_code]
        #             y = torch.FloatTensor([get_label(pdb_code, label_df)])
        #             data = Data(node_feats, edge_index, edge_feats, y=y, pos=pos)
        #             data.pdb = pdb_code
        #             torch.save(data, os.path.join(self.processed_dir, 'data_{}.pt'.format(num_codes)))
        #             num_codes += 1
        # print('Num codes', num_codes)
        # print('Num pairs', num_pairs)

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
    if split_file is None:
        return DataLoader(dataset, batch_size, shuffle=True)
    indices = sp.read_split_file(split_file)

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

    print(len(data))
    outfile = open(os.path.join(out_dir, '{}-to-{}_graph.pkl'.format(target, start)), 'wb')
    pickle.dump(data, outfile)
    print(error_count)

def get_unfinished_prots(fname, out_dir):
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

def get_prots(fname):
    pairs = []
    with open(fname) as fp:
        for line in tqdm(fp, desc='files'):
            if line[0] == '#': continue
            protein, target, start = line.strip().split()
            pairs.append((protein, target, start))

    return pairs

def split_process(target, start, start_num_code):
    label_df = pd.read_csv(label_file)
    graph_dir = '{}/{}-to-{}_graph.pkl'.format(graph_root, target, start)
    infile = open(graph_dir, 'rb')
    graph_data = pickle.load(infile)
    infile.close()
    for pdb_code in tqdm(graph_data, desc='pdb_codes'):
        node_feats, edge_index, edge_feats, pos = graph_data[pdb_code]
        y = torch.FloatTensor([get_label(pdb_code, label_df)])
        data = Data(node_feats, edge_index, edge_feats, y=y, pos=pos)
        data.pdb = pdb_code
        torch.save(data, os.path.join(processed_root, 'data_{}.pt'.format(start_num_code)))
        start_num_code += 1

def get_code_groups(fname):
    pkl_file = '/home/users/sidhikab/lig_clash_score/src/examples/pytorch_geometric/pairs.pkl'
    if (os.path.exists(pkl_file)):
        infile = open(pkl_file, 'rb')
        pairs = pickle.load(infile)
        infile.close()
        return pairs
    else:
        label_df = pd.read_csv(label_file)
        pairs = []
        num_pairs = 0
        num_codes = 0

        with open(fname) as fp:
            for line in tqdm(fp, desc='files'):
                if line[0] == '#': continue
                if num_codes / len(label_df) > CUTOFF:
                    break
                protein, target, start = line.strip().split()
                num_pairs += 1
                pairs.append((protein, target, start, num_codes))
                graph_dir = '{}/{}-to-{}_graph.pkl'.format(graph_root, target, start)
                infile = open(graph_dir, 'rb')
                graph_data = pickle.load(infile)
                infile.close()
                for _ in graph_data:
                    num_codes += 1

        outfile = open(pkl_file, 'wb')
        pickle.dump(pairs, outfile)
        print(num_codes)
        print(num_pairs)

        return pairs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either all or group')
    parser.add_argument('root', type=str, help='either all or group')
    parser.add_argument('out_dir', type=str, help='either all or group')
    parser.add_argument('prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('--group', type=int, default=-1, help='if type is group, argument indicates group index')
    parser.add_argument('--target', type=str, default='', help='if type is group, argument indicates group index')
    parser.add_argument('--start', type=str, default='', help='if type is group, argument indicates group index')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    if args.task == 'all':
        pairs = get_prots(args.prot_file)
        n = 3
        grouped_files = []

        for i in range(0, len(pairs), n):
            grouped_files += [pairs[i: i + n]]
        # for i in range(len(grouped_files)):
        #     cmd = 'sbatch -p owners -t 5:00:00 -o {} --wrap="' \
        #           '/home/groups/rondror/software/sidhikab/miniconda/envs/geometric/bin/python pdbbind_dataloader.py group ' \
        #           '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d ' \
        #           '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/graphs ' \
        #           '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt --group {}"'
        #     os.system(cmd.format(os.path.join(run_path, 'graph_{}.out'.format(i)), i))
        #     # print(cmd.format(os.path.join(run_path, 'graph_{}.out'.format(i)), i))
        print(len(grouped_files))

    if args.task == 'group':
        pairs = get_prots(args.prot_file)
        n = 3
        grouped_files = []

        for i in range(0, len(pairs), n):
            grouped_files += [pairs[i: i + n]]
        for _, target, start in grouped_files[args.group]:
            create_graphs(target, start, args.root, args.out_dir)

    if args.task == 'pdb':
        target, start = args.target, args.start
        create_graphs(target, start, args.root, args.out_dir)

    if args.task == 'check':
        pairs, unfinished_pairs = get_unfinished_prots(args.prot_file, args.out_dir)
        print('Missing:', len(unfinished_pairs), '/', len(pairs))
        # print(unfinished_pairs)

    if args.task == 'combine_all':
        pairs = get_code_groups(args.prot_file)
        n = 1
        grouped_files = []
        print(len(grouped_files))

        for i in range(0, len(pairs), n):
            grouped_files += [pairs[i: i + n]]
        for i in range(len(grouped_files)):
            cmd = 'sbatch -p owners -t 5:00:00 -o {} --wrap="' \
                  '/home/groups/rondror/software/sidhikab/miniconda/envs/geometric/bin/python pdbbind_dataloader.py combine_group ' \
                  '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d ' \
                  '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/graphs ' \
                  '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt --group {}"'
            os.system(cmd.format(os.path.join(run_path, 'graph_{}.out'.format(i)), i))
            # print(cmd.format(os.path.join(run_path, 'graph_{}.out'.format(i)), i))
        print(len(grouped_files))

    if args.task == 'combine_group':
        pairs = get_code_groups(args.prot_file)
        n = 1
        grouped_files = []

        for i in range(0, len(pairs), n):
            grouped_files += [pairs[i: i + n]]

        for _, target, start, num_codes in grouped_files[args.group]:
            split_process(target, start, num_codes)

    if args.task == 'combine_check':
        label_df = pd.read_csv(label_file)
        unfinished_pairs = []
        num_codes = 0
        num_pairs = 0

        with open(args.prot_file) as fp:
            for line in tqdm(fp, desc='files'):
                if line[0] == '#': continue
                protein, target, start = line.strip().split()
                if num_codes / len(label_df) > CUTOFF:
                    break
                num_pairs += 1
                graph_dir = '{}/{}-to-{}_graph.pkl'.format(graph_root, target, start)
                infile = open(graph_dir, 'rb')
                graph_data = pickle.load(infile)
                infile.close()
                start_num_code = num_codes
                for _ in graph_data:
                    if not os.path.exists(os.path.join(processed_root, 'data_{}.pt'.format(num_codes))):
                        unfinished_pairs.append((protein, target, start, start_num_code))
                        break
                    num_codes += 1

        print('Missing', len(unfinished_pairs), '/', num_pairs)
        print(unfinished_pairs)

    if args.task == 'pdbbind_dataloader':
        data = pdbbind_dataloader(1, data_dir='/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d')
        print(len(data))

if __name__=="__main__":
    main()



