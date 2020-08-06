'''
This protocol can be used to convert pdb files to mae files

how to run this file:
ml load chemistry
ml load schrodinger
/home/groups/rondror/software/sidhikab/miniconda/envs/geometric/bin/python lig_prot_combiner.py /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt
'''

import argparse
import os

raw_root = '/oak/stanford/groups/rondror/projects/ligand-docking/pdbbind_2019/refined-set-raw'
data_root = '/oak/stanford/groups/rondror/projects/ligand-docking/pdbbind_2019/data'
run_path = '/home/users/sidhikab/flexibility_project/atom3d/src/data_analysis/run'


'''
Get the amino acid sequence
:param file: .mae file for the structure
:return: the amino acid string for all amino acids in chain A
'''
def find_missing(file):
    ls = []
    with open(file) as fp:
        for line in fp:
            if line[0] == '#': continue
            protein, start, target = line.strip().split()
            data_protein_folder = os.path.join(data_root, protein + '/structures/aligned')
            data_file = os.path.join(data_protein_folder, start)
            ls.append((data_file))
            data_file = os.path.join(data_protein_folder, target)
            ls.append((data_file))
    return ls

def write_files(files):
    grouped_files = []
    n = 200

    for i in range(0, len(files), n):
        grouped_files += [files[i: i + n]]

    if not os.path.exists(run_path):
        os.mkdir(run_path)

    for i, group in enumerate(grouped_files):
        print('converting to pdb', i)

        with open(os.path.join(run_path, 'merge{}_in.sh'.format(i)), 'w') as f:
            for output_folder in group:
                f.write('#!/bin/bash\n')
                f.write('cat {}_prot.mae {}_lig.mae > {}_merge_pv.mae \n'.format(output_folder, output_folder, output_folder))
                f.write('$SCHRODINGER/run pv_convert.py -mode merge {}_merge_pv.mae \n'.format(output_folder))
                f.write('rm {}_merge_pv.mae\n'.format(output_folder))

        os.chdir(run_path)
        os.system('sbatch -p rondror -t 02:00:00 -o merge{}.out merge{}_in.sh'.format(i, i))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    args = parser.parse_args()

    files = find_missing(args.docked_prot_file)
    write_files(files)

if __name__ == '__main__':
    main()
