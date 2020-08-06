"""
The purpose of this code is to label the data
It can be run on sherlock using
ml load chemistry
ml load schrodinger
/home/groups/rondror/software/sidhikab/miniconda/envs/geometric/bin/python decoy_rmsd.py run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt
/home/groups/rondror/software/sidhikab/miniconda/envs/geometric/bin/python decoy_rmsd.py check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt

"""

#!/usr/bin/env python
# coding: utf-8

import os
import argparse
from tqdm import tqdm
import pickle

PROTEIN_CUTOFF = 2000
run_path = '/home/users/sidhikab/flexibility_project/atom3d/src/data/run'

def get_prots(fname):
    pairs = []
    with open(fname) as fp:
        for line in tqdm(fp, desc='files'):
            if line[0] == '#': continue
            protein, target, start = line.strip().split()
            pairs.append((protein, target, start))

    return pairs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either run or check')
    parser.add_argument('datapath', type=str, help='directory where PDBBind is located')
    parser.add_argument('prot_file', type=str, help='file listing proteins to process')
    args = parser.parse_args()

    if not os.path.exists(run_path):
        os.mkdir(run_path)

    pairs = get_prots(args.prot_file)

    if args.task == 'run':
        n = 3
        grouped_files = []

        for i in range(0, len(pairs), n):
            grouped_files += [pairs[i: i + n]]

        for i in range(len(grouped_files)):
            with open(os.path.join(run_path, 'rmsd{}_in.sh'.format(i)), 'w') as f:
                f.write('#!/bin/bash\n')
                for protein, target, start in grouped_files[i]:
                    print(protein, target, start)
                    pair_path = os.path.join(args.datapath, '{}/{}-to-{}'.format(protein, target, start))
                    f.write('cd {}\n'.format(pair_path))
                    files = []
                    for file in os.listdir(os.path.join(args.datapath, pair_path)):
                        if '{}_lig'.format(target) in file and file[-3:] == 'mae':
                            files.append(file)
                    f.write('cat {}_prot.mae {} > {}-to-{}_merge_pv.mae\n'.format(start, ' '.join(files), target, start))
                    f.write('$SCHRODINGER/run rmsd.py -use_neutral_scaffold -pv second -c {}-to-{}_rmsd.out {}_lig0.mae {}-to-{}_merge_pv.mae\n'.format(target, start, target, target, start))
                    with open(os.path.join(pair_path, '{}-to-{}_rmsd_index.pkl'.format(target, start)), 'wb') as pickle_f:
                        pickle.dump(files, pickle_f)
            os.chdir(run_path)
            os.system('sbatch -p owners -t 02:00:00 -o rmsd{}.out rmsd{}_in.sh'.format(i, i))

        print(len(grouped_files))

    if args.task == 'check':
        unfinished = []
        for protein, target, start in pairs:
            pair_path = os.path.join(args.datapath, '{}/{}-to-{}'.format(protein, target, start))
            if not os.path.exists(os.path.join(pair_path, '{}-to-{}_rmsd.out'.format(target, start))):
                unfinished.append((protein, target, start))

        print('Missing', len(unfinished), '/', len(pairs))
        # print(unfinished)

if __name__ == "__main__":
    main()