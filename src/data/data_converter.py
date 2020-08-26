'''
This protocol can be used to convert protein mae files to pdb files or ligand mae files to sdf files

how to run this file:
ml load chemistry
ml load schrodinger
$ $SCHRODINGER/run python3 data_converter.py run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
$ $SCHRODINGER/run python3 data_converter.py check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
$ $SCHRODINGER/run python3 data_converter.py remove_pv /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
$ $SCHRODINGER/run python3 data_converter.py MAPK14
'''

import os
from tqdm import tqdm
import argparse
import schrodinger.structure as structure

N = 4000
MAX_POSES = 100
MAX_DECOYS = 10

def find_files(docked_prot_file, raw_root):
    '''
    Get the files for all protein, target, start groups
    :param docked_prot_file: (string) file listing proteins to process
    :param raw_root: (string) path to raw data directory
    :return: process (list) list of all files to convert
    '''
    process = []
    with open(docked_prot_file) as fp:
        for line in tqdm(fp, desc='going through protein, target, start groups'):
            if line[0] == '#': continue
            protein, target, start = line.strip().split()
            protein_path = os.path.join(raw_root, protein)
            pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
            pose_path = os.path.join(pair_path, 'ligand_poses')

            # stage basic files
            for file in os.listdir(pair_path):
                if file == 'ligand_poses':
                    continue
                if 'prot' in file and not os.path.exists(os.path.join(pair_path, file)[:-4] + '.pdb'):
                    process.append(os.path.join(pair_path, file)[:-4])
                if 'lig' in file and not os.path.exists(os.path.join(pair_path, file)[:-4] + '.sdf'):
                    process.append(os.path.join(pair_path, file)[:-4])

            # stage ligand pose files
            for file in os.listdir(pose_path):
                if 'lig' in file and not os.path.exists(os.path.join(pose_path, file)[:-4] + '.sdf'):
                    process.append(os.path.join(pose_path, file)[:-4])



    return process

def find_MAPK14_files(save_root):
    '''
    Get the files for all MAPK14 target, start groups=
    :return: ls (list) list of all files to convert
    '''
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

def write_files(files, run_path):
    '''
    Writes a script to convert batches of files
    :param: files (list) list of all files to convert
    :param: run_path (string) path to directory where scripts and outputs will be written
    :return:
    '''
    grouped_files = group_files(N, files)

    if not os.path.exists(run_path):
        os.mkdir(run_path)

    for i, group in enumerate(grouped_files):
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
        os.system('sbatch -p owners -t 02:00:00 -o convert{}.out convert{}_in.sh'.format(i, i))
        # print('sbatch -p owners -t 02:00:00 -o convert{}.out convert{}_in.sh'.format(i, i))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either run, check, remove_pv, or MAPK14')
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    args = parser.parse_args()

    if args.task == 'run':
        files = find_files(args.docked_prot_file, args.raw_root)
        write_files(files, args.run_path)

    if args.task == 'check':
        process = []
        num_pairs = 0
        with open(args.docked_prot_file) as fp:
            for line in tqdm(fp, desc='going through protein, target, start groups'):
                if line[0] == '#': continue
                protein, target, start = line.strip().split()
                num_pairs += 1
                protein_path = os.path.join(args.raw_root, protein)
                pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
                pose_path = os.path.join(pair_path, 'ligand_poses')

                # check basic files
                if not os.path.exists('{}/{}_prot.pdb'.format(pair_path, start)) or not os.path.exists(
                        '{}/{}_lig.sdf'.format(pair_path, start)):
                    process.append((protein, start, target))
                    continue

                # check ligand pose files
                pv_file = os.path.join(pair_path, '{}-to-{}_pv.maegz'.format(target, start))
                # num_poses = min(MAX_POSES, len(list(structure.StructureReader(pv_file))))
                num_poses = 0
                for i in range(MAX_DECOYS):
                    if not os.path.join(pose_path, '{}_lig{}.sdf'.format(target, str(num_poses) + chr(ord('a') + i))):
                        process.append((protein, target, start))
                        break

        print('Missing', len(process), '/', num_pairs)
        print(process)

    elif args.task == 'MAPK14':
        files = find_MAPK14_files(args.raw_root)
        write_files(files, args.run_path)

if __name__ == '__main__':
    main()