'''
This protocol can be used to convert pdb files to mae files

how to run this file:
ml load chemistry
ml load schrodinger
/home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python data_converter.py regular
/home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python data_converter.py MAPK14
'''

import os
from tqdm import tqdm
import argparse

save_root = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw'
run_path = '/home/users/sidhikab/lig_clash_score/src/data/run'


'''
Get the amino acid sequence
:param file: .mae file for the structure
:return: the amino acid string for all amino acids in chain A
'''
def find_files():
    ls = []
    for protein in tqdm(os.listdir(save_root), desc='protein directories'):
        protein_root = os.path.join(save_root, protein)
        for pair in os.listdir(protein_root):
            pair_root = os.path.join(protein_root, pair)
            for file in os.listdir(pair_root):
                if 'prot' in file and not os.path.exists(os.path.join(pair_root, file)[:-4] + '.pdb'):
                    ls.append(os.path.join(pair_root, file)[:-4])
                if 'lig' in file and not os.path.exists(os.path.join(pair_root, file)[:-4] + '.sdf'):
                    ls.append(os.path.join(pair_root, file)[:-4])
    return ls

def find_MAPK14_files():
    ls = []
    protein = 'MAPK14'
    protein_root = os.path.join(save_root, protein)
    for pair in os.listdir(protein_root):
        pair_root = os.path.join(protein_root, pair)
        for file in tqdm(os.listdir(pair_root), desc='files in protein directory'):
            if 'prot' in file and not os.path.exists(os.path.join(pair_root, file)[:-4] + '.pdb'):
                ls.append(os.path.join(pair_root, file)[:-4])
            if 'lig' in file and not os.path.exists(os.path.join(pair_root, file)[:-4] + '.sdf'):
                ls.append(os.path.join(pair_root, file)[:-4])
    return ls

def write_files(files):
    grouped_files = []
    n = 4000

    for i in range(0, len(files), n):
        grouped_files += [files[i: i + n]]

    if not os.path.exists(run_path):
        os.mkdir(run_path)

    for i, group in enumerate(grouped_files):

        print('converting', i)
        if i == 100:
            break

        with open(os.path.join(run_path, 'convert{}_in.sh'.format(i)), 'w') as f:
            f.write('#!/bin/bash\n')
            for file in group:
                if 'prot' in file:
                    f.write('$SCHRODINGER/utilities/structconvert -imae {}.mae -opdb {}.pdb \n'.format(
                        file, file))
                else:
                    f.write('$SCHRODINGER/utilities/sdconvert -imae {}.mae -osd {}.sdf \n'.format(
                        file, file))

        os.chdir(run_path)
        os.system('sbatch -p rondror -t 02:00:00 -o convert{}.out convert{}_in.sh'.format(i, i))
        # print('sbatch -p owners -t 02:00:00 -o convert{}.out convert{}_in.sh'.format(i, i))

"""
gets list of all protein, target ligands, and starting ligands in the index file
:param docked_prot_file: (string) file listing proteins to process
:return: process (list) list of all protein, target ligands, and starting ligands to process
"""
def get_prots(docked_prot_file):
    process = []
    with open(docked_prot_file) as fp:
        for line in fp:
            if line[0] == '#': continue
            protein, target, start = line.strip().split()
            process.append((protein, target, start))

    return process

"""
groups pairs into sublists of size n
:param n: (int) sublist size
:param process: (list) list of pairs to process
:return: grouped_files (list) list of sublists of pairs
"""
def group_files(n, process):
    grouped_files = []
     cvvbfvvcfvvvv

    for i in range(0, len(process), n):
        grouped_files += [process[i: i + n]]

    return grouped_files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either all, group, check, or MAPK14')
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('--index', type=int, default=-1, help='for group task, group number')
    args = parser.parse_args()

    if args.task == 'all':
        process = get_prots(args.docked_prot_file)
        grouped_files = group_files(N, process)

        if not os.path.exists(args.run_path):
            os.mkdir(args.run_path)

        for i, group in enumerate(grouped_files):
            cmd = 'sbatch -p owners -t 1:00:00 -o {} --wrap="$SCHRODINGER/run python3 decoy_creator.py group {} {} {} ' \
                  '--index {}"'
            os.system(cmd.format(os.path.join(args.run_path, 'decoy{}.out'.format(i)), args.docked_prot_file,
                                 args.run_path, args.raw_root, i))

    if args.task == 'group':
        process = get_prots(args.docked_prot_file)
        grouped_files = group_files(N, process)

        for protein, target, start in grouped_files[args.index]:
            protein_path = os.path.join(args.raw_root, protein)
            pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
            pose_path = os.path.join(pair_path, 'ligand_poses')
            pv_file = os.path.join(pair_path, '{}-to-{}_pv.maegz'.format(target, start))
            num_poses = len(list(structure.StructureReader(pv_file)))
            for i in range(1, num_poses):
                if i == MAX_POSES:
                    break
                lig_file = os.path.join(pose_path, '{}_lig{}.mae'.format(target, i))
                create_decoys(lig_file)

    elif args.task == 'MAPK14':
        files = find_MAPK14_files()
        write_files(files)

if __name__ == '__main__':
    main()