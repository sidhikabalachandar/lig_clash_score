"""
The purpose of this code is to set the train, val, and test data sets
It can be run on sherlock using
$ /home/groups/rondror/software/sidhikab/miniconda/envs/atom3d/bin/python split_data.py /home/users/sidhikab/flexibility_project/atom3d/Data/refined_random.txt
"""

import argparse
import random
import os

TEST_PROP = 0.15
SEED = 0
save_path = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits'

def read_file(f):
    proteins = {}
    num_prots = 0
    with open(f) as fp:
        for line in fp:
            if line[0] == '#': continue
            protein, start, target = line.strip().split()
            if protein not in proteins:
                proteins[protein] = {}
            if start not in proteins[protein]:
                proteins[protein][start] = []
            proteins[protein][start].append(target)
            num_prots += 1

    return proteins, num_prots

def split_data(proteins, num_prots):
    prot_names = list(proteins.keys())
    random.shuffle(prot_names)
    train_proteins = {}
    val_proteins = {}
    test_proteins = {}
    counter = 0
    num_train = 0
    num_test = 0
    num_val = 0
    for prot in prot_names:
        if counter < num_prots * TEST_PROP:
            for start in proteins[prot]:
                counter += len(proteins[prot][start])
                if prot not in proteins:
                    test_proteins[prot] = {}
                if start not in proteins[prot]:
                    test_proteins[prot][start] = proteins[prot][start]
        elif counter < 2 * num_prots * TEST_PROP:
            if num_test == 0:
                num_test = counter
            for start in proteins[prot]:
                counter += len(proteins[prot][start])
                if prot not in proteins:
                    val_proteins[prot] = {}
                if start not in proteins[prot]:
                    val_proteins[prot][start] = proteins[prot][start]
        else:
            if num_val == 0:
                num_val = counter - num_test
            for start in proteins[prot]:
                num_train += len(proteins[prot][start])
                if prot not in proteins:
                    train_proteins[prot] = {}
                if start not in proteins[prot]:
                    train_proteins[prot][start] = proteins[prot][start]

    total = num_train + num_val + num_test

    print("Num train = {}, val = {}, test = {}".format(str(num_train), str(num_val), str(num_test)))
    print("Percentage train = {}, val = {}, test = {}".format(str(num_train / total), str(num_val / total),
                                                              str(num_test / total)))
    return train_proteins, val_proteins, test_proteins

def write_file(proteins, file_name):
    text = []
    for protein in proteins:
        for start in proteins[protein]:
            for target in proteins[protein][start]:
                text.append("{} {} {}\n".format(protein, start, target))

    file = open(os.path.join(save_path, file_name), "w")
    file.writelines(text)
    file.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    args = parser.parse_args()

    random.seed(SEED)

    proteins, num_prots = read_file(args.docked_prot_file)
    train_proteins, val_proteins, test_proteins = split_data(proteins, num_prots)
    write_file(train_proteins, 'train_pdb_70_15_15.txt')
    write_file(val_proteins, 'val_pdb_70_15_15.txt')
    write_file(test_proteins, 'test_pdb_70_15_15.txt')

if __name__=="__main__":
    main()