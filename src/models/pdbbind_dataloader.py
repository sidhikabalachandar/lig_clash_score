"""
The purpose of this code is to create the pytorch-geometric graphs, create the Data files, and to load the
train/val/test data

It can be run on sherlock using
$ /home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python pdbbind_dataloader.py combine_all /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits/train_index_balance_clash.txt /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits/val_index_balance_clash.txt /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits/test_index_balance_clash.txt /home/users/sidhikab/lig_clash_score/src/models/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/processed_clustered_score/processed /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/combined.csv --score_feature
$ /home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python pdbbind_dataloader.py combine_group /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits/train_index_balance_clash.txt /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits/val_index_balance_clash.txt /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits/test_index_balance_clash.txt /home/users/sidhikab/lig_clash_score/src/models/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/processed_clustered/processed /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/combined.csv --index 0 --score_feature
$ /home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python pdbbind_dataloader.py combine_check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits/train_index_balance_clash.txt /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits/val_index_balance_clash.txt /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits/test_index_balance_clash.txt /home/users/sidhikab/lig_clash_score/src/models/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/processed_clustered/processed /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/combined.csv --score_feature
$ /home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python pdbbind_dataloader.py pdbbind_dataloader /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits/train_index_balance_clash.txt /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits/val_index_balance_clash.txt /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits/test_index_balance_clash.txt /home/users/sidhikab/lig_clash_score/src/models/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/processed_clustered_score/processed /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/combined.csv --score_feature
"""

import sys
sys.path[-2] = '/home/users/sidhikab/lig_clash_score/src'
from util import splits as sp

import pandas as pd
import os
import torch
from torch_geometric.data import Dataset, Data, DataLoader
from tqdm import tqdm
import argparse
import pickle
import random

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

def get_index_groups(process, raw_root):
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
    index_groups = []
    num_codes = 0
    for protein, target, start in tqdm(process, desc='going through protein, target, start groups'):
        index_groups.append((protein, target, start, num_codes))

        #update num_codes
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
        cluster_dir = os.path.join(pair_path, '{}-to-{}_clustered.pkl'.format(target, start))
        infile = open(cluster_dir, 'rb')
        cluster_data = pickle.load(infile)
        infile.close()
        num_codes += len(cluster_data)

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

def get_score_no_vdw(pdb, label_df):
    """
    searches for pdb's rmsd in combined rmsd df
    :param pdb: (string) {target}_lig{id}
    :param label_df: (df) combined rmsd df
    :return: (float) rmsd value
    """
    return label_df[label_df['target'] == pdb]['score_no_vdw'].iloc[0]

def split_process(protein, target, start, label_file, pair_path, processed_root, start_index, include_score, lower_score_bound, upper_score_bound):
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
    if not include_score:
        graph_dir = '{}/{}-to-{}_graph.pkl'.format(pair_path, target, start)
    else:
        graph_dir = '{}/{}-to-{}_graph_with_score.pkl'.format(pair_path, target, start)
    infile = open(graph_dir, 'rb')
    graph_data = pickle.load(infile)
    infile.close()
    cluster_dir = os.path.join(pair_path, '{}-to-{}_clustered.pkl'.format(target, start))
    infile = open(cluster_dir, 'rb')
    cluster_data = pickle.load(infile)
    infile.close()
    print(len(cluster_data))

    for pdb_code in tqdm(cluster_data, desc='pdb_codes'):
        node_feats, edge_index, edge_feats, pos = graph_data[pdb_code]
        y = torch.FloatTensor([get_label(pdb_code, label_df)])
        data = Data(node_feats, edge_index, edge_feats, y=y, pos=pos)
        data.pdb = '{}_{}-to-{}_{}'.format(protein, target, start, pdb_code)
        score = get_score_no_vdw(pdb_code, label_df)
        if score < lower_score_bound:
            score = lower_score_bound
        if score > upper_score_bound:
            score = upper_score_bound
        data.physics_score = score
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

def run_combine_all(train_prot_file, val_prot_file, test_prot_file, run_path, root, processed_root, label_file,
                    grouped_files, n, include_score):
    for i, group in enumerate(grouped_files):
        if include_score:
            cmd = 'sbatch -p owners -t 1:00:00 -o {} --wrap="' \
                  '/home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python pdbbind_dataloader.py ' \
                  'combine_group {} {} {} {} {} {} {} --score_feature --n {} --index {}"'
        else:
            cmd = 'sbatch -p owners -t 1:00:00 -o {} --wrap="' \
                  '/home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python pdbbind_dataloader.py ' \
                  'combine_group {} {} {} {} {} {} {} --n {} --index {}"'
        os.system(cmd.format(os.path.join(run_path, 'processed{}.out'.format(i)), train_prot_file, val_prot_file,
                             test_prot_file, run_path, root, processed_root, label_file, n, i))

def run_combine_group(grouped_files, raw_root, processed_root, label_file, index, include_score, lower_score_bound,
                      upper_score_bound):
    for protein, target, start, start_index in grouped_files[index]:
        print(protein, target, start)
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
        split_process(protein, target, start, label_file, pair_path, processed_root, start_index, include_score, lower_score_bound, upper_score_bound)

def run_combine_check(process, raw_root, processed_root):
    num_codes = 0
    index_groups = []
    for protein, target, start in tqdm(process, desc='going through protein, target, start groups'):
        # update num_codes
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
        cluster_dir = os.path.join(pair_path, '{}-to-{}_clustered.pkl'.format(target, start))
        infile = open(cluster_dir, 'rb')
        cluster_data = pickle.load(infile)
        infile.close()

        for _ in cluster_data:
            if not os.path.exists(os.path.join(processed_root, 'data_{}.pt'.format(num_codes))):
                index_groups.append((protein, target, start, num_codes))
                break
            num_codes += 1

    print('Missing', len(index_groups), '/', len(process))
    print(index_groups)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either all, group, check, MAPK14, combine_all, combine_group, '
                                               'combine_check, or pdbbind_dataloader')
    parser.add_argument('train_prot_file', type=str, help='file listing proteins to process for training dataset')
    parser.add_argument('val_prot_file', type=str, help='file listing proteins to process for validation dataset')
    parser.add_argument('test_prot_file', type=str, help='file listing proteins to process for testing dataset')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('root', type=str, help='directory where raw and processed directories can be found')
    parser.add_argument('processed_root', type=str, help='directory where raw and processed directories can be found')
    parser.add_argument('label_file', type=str, help='file with rmsd labels')
    parser.add_argument('--index', type=int, default=-1, help='for group task, group number')
    parser.add_argument('--n', type=int, default=3, help='number of protein, target, start groups processed in '
                                                         'group task')
    parser.add_argument('--lower_score_bound', type=float, default=-20, help='any physics score below this value, will '
                                                                             'be set to this value')
    parser.add_argument('--upper_score_bound', type=float, default=20, help='any physics score above this value, will '
                                                                            'be set to this value')
    parser.add_argument('--score_feature', dest='include_score', action='store_true')
    parser.add_argument('--no_score_feature', dest='include_score', action='store_false')
    parser.set_defaults(include_score=False)
    args = parser.parse_args()

    raw_root = os.path.join(args.root, 'raw')
    random.seed(0)

    if not os.path.exists(args.processed_root):
        os.mkdir(args.processed_root)

    if not os.path.exists(args.run_path):
        os.mkdir(args.run_path)

    if args.task == 'combine_all':
        process = get_prots(args.train_prot_file)
        process.extend(get_prots(args.val_prot_file))
        process.extend(get_prots(args.test_prot_file))
        index_groups = get_index_groups(process, raw_root)
        grouped_files = group_files(args.n, index_groups)
        run_combine_all(args.train_prot_file, args.val_prot_file, args.test_prot_file, args.run_path, args.root,
                        args.processed_root, args.label_file, grouped_files, args.n, args.include_score)

    if args.task == 'combine_group':
        process = get_prots(args.train_prot_file)
        process.extend(get_prots(args.val_prot_file))
        process.extend(get_prots(args.test_prot_file))
        index_groups = get_index_groups(process, raw_root)
        grouped_files = group_files(args.n, index_groups)
        run_combine_group(grouped_files, raw_root, args.processed_root, args.label_file, args.index, args.include_score,
                          args.lower_score_bound, args.upper_score_bound)

    if args.task == 'combine_check':
        process = get_prots(args.train_prot_file)
        process.extend(get_prots(args.val_prot_file))
        process.extend(get_prots(args.test_prot_file))
        run_combine_check(process, raw_root, args.processed_root)

    if args.task == 'pdbbind_dataloader':
        if not args.include_score:
            loader = pdbbind_dataloader(1, "/home/users/sidhikab/lig_clash_score/src/models/out/test.out", data_dir=args.root)
        else:
            loader = pdbbind_dataloader(1, "/home/users/sidhikab/lig_clash_score/src/models/out/test.out",
                                        data_dir=os.path.join(args.root, 'processed_score'))
        print(len(loader))
        for data in loader:
            print(data)
            print(data.pdb)
            break

if __name__=="__main__":
    main()



