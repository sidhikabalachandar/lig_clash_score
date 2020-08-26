"""
The purpose of this code is to create the rmsd csv files, combine them into one csv file, and search in this csv file

It can be run on sherlock using
ml load chemistry
ml load schrodinger
/home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python get_labels.py all /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/atom3d/protein_ligand/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d
/home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python get_labels.py group /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/atom3d/protein_ligand/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d --index 0
/home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python get_labels.py check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/atom3d/protein_ligand/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d
/home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python get_labels.py combine /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/atom3d/protein_ligand/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d
/home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python get_labels.py MAPK14
"""

import pandas as pd
import os
import argparse
from tqdm import tqdm
import pickle

N = 30

def get_label(pdb, label_df):
    """
    searches for pdb's rmsd in combined rmsd df
    :param pdb: (string) {target}_lig{id}
    :param label_df: (df) combined rmsd df
    :return: (float) rmsd value
    """
    return label_df[label_df['target'] == pdb]['rmsd'].iloc[0]

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

def to_df(data, out_dir, pair):
    '''
    write list of rmsds to csv file
    :param data: (list) list of rmsds
    :param out_dir: (string) path to output directory for this pair's csv file
    :param pair: (string) {target}-to-{start}
    :return: all_rmsds: (list) list of all rmsds
    '''
    df = pd.DataFrame(data, columns=['protein', 'start', 'target', 'rmsd'])
    df.to_csv(os.path.join(out_dir, pair + '_rmsd.csv'))

def get_rmsd_results(rmsd_file_path):
    '''
    Get list of rmsds
    Format of csv file:
    "Index","Title","Mode","RMSD","Max dist.","Max dist atom index pair","ASL"
    "1","2W1I_pdb.smi:1","In-place","8.38264277875","11.5320975551","10-20","not atom.element H"
    :param rmsd_file_path: (string) rmsd file
    :return: all_rmsds: (list) list of all rmsds
    '''
    all_rmsds = []
    with open(rmsd_file_path) as rmsd_file:
        for line in list(rmsd_file)[1:]:
            rmsd = line.split(',')[3].strip('"')
            all_rmsds.append(float(rmsd))
    return all_rmsds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either all, group, check, combine, or MAPK14')
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('out_dir', type=str, help='directory where the combined rmsd csv file will be placed')
    parser.add_argument('--index', type=int, default=-1, help='for group task, group number')
    args = parser.parse_args()

    if args.task == 'all':
        process = get_prots(args.docked_prot_file)
        grouped_files = group_files(N, process)

        if not os.path.exists(args.run_path):
            os.mkdir(args.run_path)

        for i, group in enumerate(grouped_files):
            cmd = 'sbatch -p owners -t 1:00:00 -o {} --wrap="' \
                  '/home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python get_labels.py group ' \
                  '{} {} {} {} --index {}"'
            os.system(cmd.format(os.path.join(args.run_path, 'labels{}.out'.format(i)), args.docked_prot_file,
                                 args.run_path, args.raw_root, args.out_dir, i))

    if args.task == 'group':
        process = get_prots(args.docked_prot_file)
        grouped_files = group_files(N, process)

        for protein, target, start in grouped_files[args.index]:
            protein_path = os.path.join(args.raw_root, protein)
            pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
            # infile = open(os.path.join(pair_path, '{}-to-{}_rmsd_index.pkl'.format(target, start)), 'rb')
            # files = pickle.load(infile)
            # infile.close()
            # rmsds = get_rmsd_results(os.path.join(pair_path, '{}-to-{}_rmsd.out'.format(target, start)))
            # pair_data = [[protein, start, files[i][:-4], rmsds[i]] for i in range(len(rmsds))]
            # to_df(pair_data, pair_path, '{}-to-{}'.format(target, start))
            os.remove(os.path.join(pair_path, '{}-to-{}_rmsd_index.pkl'.format(target, start)))
            os.remove(os.path.join(pair_path, '{}-to-{}_rmsd.out'.format(target, start)))

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
                if not os.path.exists(os.path.join(pair_path, '{}-to-{}_rmsd.csv'.format(target, start))):
                    process.append((protein, target, start))

        print('Missing', len(process), '/', num_pairs)
        print(process)

    if args.task == 'combine':
        dfs = []
        with open(args.docked_prot_file) as fp:
            for line in tqdm(fp, desc='going through protein, target, start groups'):
                if line[0] == '#': continue
                protein, target, start = line.strip().split()
                protein_path = os.path.join(args.raw_root, protein)
                pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
                dfs.append(pd.read_csv(os.path.join(pair_path, '{}-to-{}_rmsd.csv'.format(target, start))))

        combined_csv_data = pd.concat(dfs)
        combined_csv_data.to_csv(os.path.join(args.out_dir, 'pdbbind_refined_set_labels.csv'))

    if args.task == 'MAPK14':
        protein = 'MAPK14'
        ligs = ['3D83', '4F9Y']
        for target in ligs:
            for start in ligs:
                if target != start:
                    pair = '{}-to-{}'.format(target, start)
                    pair_path = os.path.join(args.datapath, '{}/{}'.format(protein, pair))
                    infile = open(os.path.join(pair_path, '{}_rmsd_index.pkl'.format(pair)), 'rb')
                    files = pickle.load(infile)
                    infile.close()
                    rmsds = get_rmsd_results(os.path.join(pair_path, '{}_rmsd.out'.format(pair)))
                    pair_data = [[protein, start.lower(), files[i][:-4].lower(), rmsds[i]] for i in range(len(rmsds))]
                    to_df(pair_data, pair_path, pair)

if __name__ == "__main__":
    main()



