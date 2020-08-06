'''
This protocol can be used to convert pdb files to mae files

how to run this file:
ml load chemistry
ml load schrodinger
/home/groups/rondror/software/sidhikab/miniconda/envs/geometric/bin/python data_converter.py
'''

import os
from tqdm import tqdm

save_root = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw'
run_path = '/home/users/sidhikab/flexibility_project/atom3d/src/data/run'


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

def main():
    files = find_files()
    print(len(files))
    # write_files(files)

if __name__ == '__main__':
    main()