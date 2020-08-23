'''
This protocol can be used to convert pdb files to mae files

how to run this file:
ml load chemistry
ml load schrodinger
/home/groups/rondror/software/sidhikab/miniconda/envs/geometric/bin/python mae_to_pdb.py /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random_with_unaligned.txt
'''

import argparse
import os

data_root = '/oak/stanford/groups/rondror/projects/ligand-docking/pdbbind_2019/data'
run_path = '/home/users/sidhikab/flexibility_project/atom3d/src/data/run'


'''
Get the amino acid sequence
:param file: .mae file for the structure
:return: the amino acid string for all amino acids in chain A
'''
def find_files(file):
    ls = []
    with open(file) as fp:
        for line in fp:
            if line[0] == '#': continue
            protein, start, target = line.strip().split()

            data_protein_folder = os.path.join(data_root, protein + '/structures/aligned')

            struct_file = os.path.join(data_protein_folder, start + '_prot.mae')
            lig_file = os.path.join(data_protein_folder, target + '_lig.mae')
            if struct_file not in ls:
                ls.append((struct_file[:-4]))
            if not os.path.exists(lig_file) and lig_file not in ls:
                ls.append((lig_file[:-4]))
    return ls

def write_files(files):
    grouped_files = []
    n = 25

    for i in range(0, len(files), n):
        grouped_files += [files[i: i + n]]

    if not os.path.exists(run_path):
        os.mkdir(run_path)

    for i, group in enumerate(grouped_files):
        print('converting to pdb', i)

        with open(os.path.join(run_path, 'grid{}_in.sh'.format(i)), 'w') as f:
            for file in group:
                f.write('#!/bin/bash\n')
                f.write('$SCHRODINGER/utilities/structconvert -imae {}.mae -opdb {}.pdb \n'.format(
                    file, file))

        os.chdir(run_path)
        os.system('sbatch -p owners -t 02:00:00 -o grid{}.out grid{}_in.sh'.format(i, i))
        # print('sbatch -p owners -t 02:00:00 -o grid{}.out grid{}_in.sh'.format(i, i))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    args = parser.parse_args()

    files = find_files(args.docked_prot_file)
    write_files(files)

if __name__ == '__main__':
    main()