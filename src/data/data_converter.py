'''
This protocol can be used to convert protein mae files to pdb files or ligand mae files to sdf files

how to run this file:
ml load chemistry
ml load schrodinger
$ $SCHRODINGER/run python3 data_converter.py run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw

$ $SCHRODINGER/run python3 data_converter.py run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits/combined_index_balance_clash_large.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw --decoy_type conformer_poses

$ $SCHRODINGER/run python3 data_converter.py check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
'''

import os
from tqdm import tqdm
import argparse
from rdkit import Chem

def get_ligand(ligfile):
    """
    Read ligand from PDB dataset into RDKit Mol. Assumes input is sdf format.
    :param ligfile: (string) ligand file
    :return: lig: (RDKit Mol object) co-crystallized ligand
    """
    lig=Chem.SDMolSupplier(ligfile)[0]
    # Many SDF in PDBBind do not parse correctly. If SDF fails, try loading the mol2 file instead
    if lig is None:
        print('trying mol2...')
        lig=Chem.MolFromMol2File(ligfile[:-4] + '.mol2')
    if lig is None:
        print('failed')
        return None
    lig = Chem.RemoveHs(lig)
    return lig

def find_files(docked_prot_file, raw_root, decoy_type):
    '''
    Get the files for all protein, target, start groups
    :param docked_prot_file: (string) file listing proteins to process
    :param raw_root: (string) path to raw data directory
    :return: process (list) list of all files to convert
    '''
    process = []
    num = 0
    with open(docked_prot_file) as fp:
        for line in tqdm(fp, desc='going through protein, target, start groups'):
            if line[0] == '#': continue
            protein, target, start = line.strip().split()
            protein_path = os.path.join(raw_root, protein)
            pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
            pose_path = os.path.join(pair_path, decoy_type)

            # stage starting protein receptor file
            process.append(os.path.join(pair_path, '{}_prot'.format(start)))

            # stage ligand pose files
            for file in os.listdir(pose_path):
                if file[-3:] == 'mae':
                    process.append(os.path.join(pose_path, file)[:-4])

            num += 1
            if num == 12:
                break

    return process

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

def write_files(files, run_path, n):
    '''
    Writes a script to convert batches of files
    :param: files (list) list of all files to convert
    :param: run_path (string) path to directory where scripts and outputs will be written
    :return:
    '''
    grouped_files = group_files(n, files)
    print(len(grouped_files))

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
                    f.write('$SCHRODINGER/utilities/sdconvert -imae {}.mae -osdf {}.sdf \n'.format(file, file))

        os.chdir(run_path)
        os.system('sbatch -p owners -t 02:00:00 -o convert{}.out convert{}_in.sh'.format(i, i))
        # print('sbatch -p owners -t 02:00:00 -o convert{}.out convert{}_in.sh'.format(i, i))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either run or check')
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('--index', type=int, default=-1, help='for group_dist_check task, group number')
    parser.add_argument('--n', type=int, default=4000, help='number of protein, target, start groups processed in '
                                                         'group task')
    parser.add_argument('--max_poses', type=int, default=100, help='maximum number of glide poses considered')
    parser.add_argument('--max_decoys', type=int, default=10, help='maximum number of decoys created per glide pose')
    parser.add_argument('--decoy_type', type=str, default='ligand_poses', help='either cartesian_poses, ligand_poses, '
                                                                               'or conformer_poses')
    args = parser.parse_args()

    if args.task == 'run':
        files = find_files(args.docked_prot_file, args.raw_root, args.decoy_type)
        write_files(files, args.run_path, args.n)

    if args.task == 'check':
        process = []
        num_pairs = 0
        with open(args.docked_prot_file) as fp:
            for line in tqdm(fp, desc='protein, target, start groups'):
                if line[0] == '#': continue
                protein, target, start = line.strip().split()
                num_pairs += 1
                protein_path = os.path.join(args.raw_root, protein)
                pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
                pose_path = os.path.join(pair_path, args.decoy_type)

                # check basic files
                if not os.path.exists('{}/{}_prot.pdb'.format(pair_path, start)):
                    process.append((protein, start, target))
                    continue

                # check ligand pose files
                for file in os.listdir(pose_path):
                    if file[-3:] == 'mae':
                        file_name = file[:-3]
                        if not os.path.exists(os.path.join(pose_path, file_name + 'sdf')):
                            process.append((protein, target, start))
                            break

        print('Missing', len(process), '/', num_pairs)
        print(process)

if __name__ == '__main__':
    main()