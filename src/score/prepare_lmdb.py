"""
The purpose of this code is to create conformers

It can be run on sherlock using
$ $SCHRODINGER/run python3 prepare_lmdb.py /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/ml_score --make_lmdb  --split
"""

#class Dataset(Dataset)
import sys
sys.path.insert(0,'/oak/stanford/groups/rondror/projects/ligand-docking/fragment_building/software/anaconda3/envs/e3nn/lib/python3.8/site-packages')

import math
import csv

import atom3d.datasets.datasets as da
import atom3d.splits.splits as spl
import collections as col
import os

from torch.utils.data import Dataset
import click

from schrodinger.structure import StructureReader


def maestruct_to_df_simple(st):
    """
    Convert a schrodinger structure object into a simple atom3D dataframe
    :param st: schrodinger structure object
    return: dataframe
    """
    df = col.defaultdict(list)
    for i, a in enumerate(st.atom):
        df['x'].append(a.x)
        df['y'].append(a.y)
        df['z'].append(a.z)
        df['element'].append(a.element.upper())
        df['serial_number'].append(a.index) # Atom index starts from 1
    df = pd.DataFrame(df)
    # Make up atom names
    return df

def file_supplier(file):
    """
    Load the file
    """
    return StructureReader(file)

def file_converter(mol):
    """
    convert the structure into atom3D format
    """
    return maestruct_to_df_simple(mol)

class MAEDataset(Dataset):
    """
    Creates a dataset from mae files that each contain a single structure.
    """

    def __init__(self, file_list, transform=None):
        """
        constructor
        :param file_list: list containing paths to mae files. Assumes one structure per file.
        :type file_list: list[Union[str, Path]]
        :param transform: transformation function for data augmentation, defaults to None
        :type transform: function, optional
        """
        self._file_list = [x for x in file_list]
        self._num_examples = len(self._file_list)
        self._transform = transform

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)
        # Read biopython structure
        file = self._file_list[index]
        mol_supplier = file_supplier(file)
        mol = next(mol_supplier)

        # assemble the item (no bonds)
        item = {
            'atoms': file_converter(mol),
            'file_path': file,
        }
        if self._transform:
            item = self._transform(item)
        return item

class MAEDataset_Grouped(Dataset):
    """
    Creates a dataset from mae files each containing MORE THAN ONE structure.
    This can be faster if you don't want to store one file per structure. 
    THe assumption is that for each file, the getitem function will be called sequentally on the structures 
    in that file.
    """

    def __init__(self, file_list, transform=None):
        """
        constructor
        :param file_list: list containing paths to mae files. Assumes one structure per file.
        :type file_list: list[Union[str, Path]]
        :param transform: transformation function for data augmentation, defaults to None
        :type transform: function, optional
        """
        self._file_list = file_list
        self._num_examples = len(file_list)
        self._transform = transform
        self._mol_supplier = None
        self._current_file = ''

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)
        # Read biopython structure
        file = self._file_list[index]
        #print("MAE get Grouped", index, file)
        if (file != self._current_file):
            self._mol_supplier = file_supplier(file)
            self._current_file = file

        mol = next(self._mol_supplier)
        #print(file, mol)
        atoms = file_converter(mol)
        # assemble the item (no bonds)
        item = {
            'atoms': atoms,
            'file_path': file,
        }
        if self._transform:
            item = self._transform(item)
        return item

class FragmentDataset(Dataset):
    """
    Base class of the fragment database
    """
    def __init__(self, input_file_path, info_list, transform = None):
        """Constructor.
        :param input_file_path: Root path to input schrodinger files. First structure
                in the file is the protein, second is the ligand
        :type input_filename: str
        :param info_list: a list of dicts of the information for each datapoint, should contain 'name', 'label', 'pocket_file', 'file'
            pocket_file and file are names of the file, they should be in input_file_path folder
        :type: list of dicts
        """
        self._ids = []
        self._labels = []
        self._load_dataset(input_file_path, info_list)
        self._num_examples = len(self._pocket_dataset)
        self._transform = transform

    def _load_dataset(self, input_file_path, info_list):
        """
        Get a list of the ligands and the corresponding pockets
        Return a list of both
        """
        file_types = ['file', 'pocket_file']
        file_lists = {}
        for f in file_types:
            file_lists[f] = []

        for info in info_list:
            self._labels.append(info['label'])
            self._ids.append(info['name'])
            for f in file_types:
                file_lists[f].append(os.path.join(input_file_path, info[f]))

        print('pocket list length', len(file_lists['pocket_file']))
        print('ligand_list first 10', file_lists['file'][0:10])

        assert len(file_lists['pocket_file']) == len(file_lists['file'])
        self._ligand_dataset = MAEDataset_Grouped(file_lists['file'])
        self._pocket_dataset = MAEDataset(file_lists['pocket_file'])

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)

        item  = {
            'atoms_ligand' : self._ligand_dataset[index]['atoms'],
            'pocket_res' : self._pocket_dataset[index]['atoms'],
            'id' : self._ids[index],
            'label' : int(self._labels[index]),
        }
        if self._transform:
            item = self._transform(item)
        return (item)

def make_lmdb_dataset(input_file_path, output_root, folder_name, ids):
    """
    Take in a list of trajectories.
    Convert them into lmdb format
    :param input_file_path: str of the file path
    :type: str
    :param output root: str of the path to the output
    :type: str
    :param ids: list information for each datapoint, a list of dicts of the information for each datapoint, should contain 'name', 'label', 'pocket_file', 'file'
            pocket_file and file are names of the file, they should be in input_file_path folder
    :type: list
    """

    lmdb_path = os.path.join(output_root, folder_name)
    os.makedirs(lmdb_path, exist_ok=True)
    if os.path.isfile(f"{output_root}/{folder_name}/data.mdb"):
        os.remove(f"{output_root}/{folder_name}/data.mdb")
        os.remove(f"{output_root}/{folder_name}/lock.mdb")

    dataset = FragmentDataset(input_file_path, ids, transform=None)
    da.make_lmdb_dataset(dataset, lmdb_path)
    return lmdb_path

def read_info_file(file_path):
    with open(file_path, "r") as f:
        a = list(csv.DictReader(f, skipinitialspace=True))
    return a


def write_split_indices(split_ids, lmdb_ds, output_txt):
    # Convert ids into lmdb numerical indices and write into txt file
    split_indices = lmdb_ds.ids_to_indices(split_ids)
    with open(output_txt, 'w') as f:
        f.write(str('\n'.join([str(i) for i in split_indices])))
    return (split_indices)

def split_lmdb_dataset(id_file_path, lmdb_path, output_folder):
    split_dir = output_folder
    logger.info(f'Splitting indices, load data from {lmdb_path:}...')
    lmdb_ds = da.load_dataset(lmdb_path, 'lmdb')
    split_ids = make_splits(id_file_path)

    logger.info(f'Write results to {split_dir:}...')
    os.makedirs(os.path.join(split_dir, 'indices'), exist_ok=True)
    os.makedirs(os.path.join(split_dir, 'data'), exist_ok=True)

    indices_train = write_split_indices(split_ids['train'], lmdb_ds, os.path.join(split_dir, 'indices/train_indices.txt'))
    indices_val = write_split_indices(split_ids['val'], lmdb_ds, os.path.join(split_dir, 'indices/val_indices.txt'))
    indices_test = write_split_indices(split_ids['test'], lmdb_ds, os.path.join(split_dir, 'indices/test_indices.txt'))

    train_dataset, val_dataset, test_dataset = spl.split(
        lmdb_ds, indices_train, indices_val, indices_test)
    da.make_lmdb_dataset(train_dataset, os.path.join(split_dir, 'data/train'))
    da.make_lmdb_dataset(val_dataset, os.path.join(split_dir, 'data/val'))
    da.make_lmdb_dataset(test_dataset, os.path.join(split_dir, 'data/test'))

import pandas as pd
def make_splits(csv_path):
    df = pd.read_csv(csv_path)
    proteins = df['pocket_file'].unique()
    N_proteins = len(proteins)
    test = math.floor(0.15*N_proteins)
    train = N_proteins - 2*test

    train_proteins = proteins[0:train]
    val_proteins = proteins[train:train+test]
    test_proteins = proteins[train+test:]

    split_ids = {'train': list(df[df.pocket_file.isin(train_proteins)].name),
        'val': list(df[df.pocket_file.isin(val_proteins)].name),
        'test': list(df[df.pocket_file.isin(test_proteins)].name)}

    return split_ids


def prepare(id_file_path, input_root, output_root, output_folder):
    info_list = read_info_file(id_file_path)
    lmdb_path = make_lmdb_dataset(input_root, output_root, output_folder, info_list)

@click.command(name='run')
@click.argument('output_root', type=click.Path())
@click.option('--make_lmdb', is_flag=True)
@click.option('--split', is_flag=True)
def run(output_root, make_lmdb, split):
    '''
    $SCHRODINGER/run python3 prepare_lmdb.py output_root --make_lmdb  --split
    
    Provide a single path 
    Under that path, have a folder called 'data' containing all input mae files
    The mae files can each contain multiple structures for the ligands
    Each pocket files should contain only 1 protein pocket

    Also that folder should have a csv called names.csv 
    That should contain the columns 'name', 'label', 'pocket_file', 'file'
    Each row corresponds to a datapoint
    'pocket_file', 'file' are the names of the mae file within the data folder containing the protein pocket and ligand structures
    '''

    input_data = f'{output_root}/data'
    name_file = 'names.csv'
    id_file_path = f"{input_data}/{name_file}"
    lmdb_folder = 'lmdb_data'
    lmdb_path = f'{output_root}/{lmdb_folder}'
    split_output_folder = f'{output_root}/{lmdb_folder}/split'

    if make_lmdb:
        prepare(id_file_path, input_data, output_root, lmdb_folder)
    if split:
        split_lmdb_dataset(id_file_path, lmdb_path, split_output_folder)



if __name__ == "__main__":
    run()
