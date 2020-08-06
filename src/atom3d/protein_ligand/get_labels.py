"""
The purpose of this code is to label the data
It can be run on sherlock using
ml load chemistry
ml load schrodinger
/home/groups/rondror/software/sidhikab/miniconda/envs/geometric/bin/python get_labels.py all /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt
/home/groups/rondror/software/sidhikab/miniconda/envs/geometric/bin/python get_labels.py check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt
/home/groups/rondror/software/sidhikab/miniconda/envs/geometric/bin/python get_labels.py combine /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt

"""

#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os
import argparse
from tqdm import tqdm
import pickle

run_path = '/home/users/sidhikab/flexibility_project/atom3d/src/atom3d/protein_ligand/run'

def get_label(pdb, label_df):
    return label_df[label_df['target'] == pdb]['rmsd'].iloc[0]

def get_prots(fname, out_dir):
    pairs = []
    unfinished_pairs = []
    with open(fname) as fp:
        for line in tqdm(fp, desc='files'):
            if line[0] == '#': continue
            protein, target, start = line.strip().split()
            pair = '{}-to-{}'.format(target, start)
            if not os.path.exists(os.path.join(out_dir, pair + '.csv')):
                unfinished_pairs.append((protein, target, start))
            pairs.append((protein, target, start))

    return pairs, unfinished_pairs

def to_df(data, out_dir, pair):
    df = pd.DataFrame(data, columns=['protein', 'start', 'target', 'rmsd'])
    df.to_csv(os.path.join(out_dir, pair + '.csv'))

def get_rmsd_results(rmsd_file_path):
    '''
    Get list of rmsds
    Format of csv file:
    "Index","Title","Mode","RMSD","Max dist.","Max dist atom index pair","ASL"
    "1","2W1I_pdb.smi:1","In-place","8.38264277875","11.5320975551","10-20","not atom.element H"
    :return:
    '''
    all_rmsds = []
    with open(rmsd_file_path) as rmsd_file:
        for line in list(rmsd_file)[1:]:
            rmsd = line.split(',')[3].strip('"')
            all_rmsds.append(float(rmsd))
    return all_rmsds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either all or group')
    parser.add_argument('datapath', type=str, help='directory where PDBBind is located')
    parser.add_argument('out_dir', type=str, help='root directory to write label files')
    parser.add_argument('prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('--group', type=int, default=-1, help='if type is group, argument indicates group index')
    args = parser.parse_args()

    indiv_labels_dir = os.path.join(args.out_dir, 'labels')

    if not os.path.exists(indiv_labels_dir):
        os.mkdir(indiv_labels_dir)

    pairs, unfinished_pairs = get_prots(args.prot_file, indiv_labels_dir)
    n = 15
    grouped_files = []

    for i in range(0, len(pairs), n):
        grouped_files += [pairs[i: i + n]]

    if args.task == 'all':
        for i in range(len(grouped_files)):
            cmd = 'sbatch -p owners -t 5:00:00 -o {} --wrap="' \
                  '/home/groups/rondror/software/sidhikab/miniconda/envs/geometric/bin/python get_labels.py group ' \
                  '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw ' \
                  '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/labels ' \
                  '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt --group {}"'
            os.system(cmd.format(os.path.join(run_path, 'label_{}.out'.format(i)), i))
            # print(cmd.format(os.path.join(run_path, 'label{}.out'.format(i)), i))
        print(len(grouped_files))

    if args.task == 'group':
        for protein, target, start in grouped_files[args.group]:
            print(protein, target, start)
            pair = '{}-to-{}'.format(target, start)
            pair_path = os.path.join(args.datapath, '{}/{}'.format(protein, pair))
            infile = open(os.path.join(pair_path, '{}_rmsd_index.pkl'.format(pair)), 'rb')
            files = pickle.load(infile)
            infile.close()
            rmsds = get_rmsd_results(os.path.join(pair_path, '{}_rmsd.out'.format(pair)))
            pair_data = [[protein, start, files[i][:-4], rmsds[i]] for i in range(len(rmsds))]
            to_df(pair_data, indiv_labels_dir, pair)

    if args.task == 'check':
        print('Missing:', len(unfinished_pairs), '/', len(pairs))
        # print(unfinished_pairs)

    if args.task == 'combine':
        combined_csv_data = pd.concat([pd.read_csv(os.path.join(indiv_labels_dir, f)) for f in os.listdir(indiv_labels_dir)])
        combined_csv_data.to_csv(os.path.join(args.out_dir, 'pdbbind_refined_set_labels.csv'))

if __name__ == "__main__":
    main()



