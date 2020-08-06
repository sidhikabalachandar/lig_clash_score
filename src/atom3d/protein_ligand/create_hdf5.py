"""
The purpose of this code is to create hdf5 files
It can be run on sherlock using
$ /home/groups/rondror/software/sidhikab/miniconda/envs/geometric/bin/python create_hdf5.py all /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/processed /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/labels /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/hdf5 /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d
/home/groups/rondror/software/sidhikab/miniconda/envs/geometric/bin/python create_hdf5.py group /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/processed /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/labels /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/hdf5 /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d --group <index>
/home/groups/rondror/software/sidhikab/miniconda/envs/geometric/bin/python create_hdf5.py check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/processed /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/labels /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/hdf5 /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d
/home/groups/rondror/software/sidhikab/miniconda/envs/geometric/bin/python create_hdf5.py combine_all /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/processed /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/labels /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/hdf5 /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d
"""

#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import numpy as np
import os
import subprocess
import sys
sys.path.append('..')
from util import datatypes as dt
from util import file as fi
from tqdm import tqdm
import argparse
from rdkit.Chem import PandasTools

run_path = '/home/users/sidhikab/flexibility_project/atom3d/src/atom3d/protein_ligand/run'

def get_prots(fname, out_dir):
    pairs = []
    unfinished_pairs = []
    with open(fname) as fp:
        for line in tqdm(fp, desc='files'):
            if line[0] == '#': continue
            protein, target, start = line.strip().split()
            pair = '{}-to-{}'.format(target, start)
            if not os.path.exists(os.path.join(out_dir, '{}_protein.csv'.format(pair)))\
                    or not os.path.exists(os.path.join(out_dir, '{}_pocket.csv'.format(pair)))\
                    or not os.path.exists(os.path.join(out_dir, '{}_pdb_codes.csv'.format(pair)))\
                    or not os.path.exists(os.path.join(out_dir, '{}.sdf'.format(pair))) \
                    or not os.path.exists(os.path.join(out_dir, '{}_labels.csv'.format(pair))):
                unfinished_pairs.append((protein, target, start))
            pairs.append((protein, target, start))

    return pairs, unfinished_pairs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either all or group')
    parser.add_argument('datapath', type=str, help='directory where data is located')
    parser.add_argument('label_root', type=str, help='path to label csv')
    parser.add_argument('out_dir', type=str, help='output file')
    parser.add_argument('prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('root', type=str, help='file listing proteins to process')
    parser.add_argument('--protein', type=str, default='None', help='if type is group, argument indicates group index')
    parser.add_argument('--target', type=str, default='None', help='if type is group, argument indicates group index')
    parser.add_argument('--start', type=str, default='None', help='if type is group, argument indicates group index')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    pairs, unfinished_pairs = get_prots(args.prot_file, args.out_dir)
    n = 5
    grouped_files = []

    for i in range(0, len(unfinished_pairs), n):
        grouped_files += [unfinished_pairs[i: i + n]]

    if args.task == 'all':
        for protein, target, start in unfinished_pairs:
            cmd = 'sbatch -p owners -t 5:00:00 -o {} --mem-per-cpu MaxMemPerCPU --wrap="' \
                  '/home/groups/rondror/software/sidhikab/miniconda/envs/geometric/bin/python create_hdf5.py group ' \
                  '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/processed ' \
                  '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/labels ' \
                  '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/hdf5 ' \
                  '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt ' \
                  '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d --protein {} --target {} --start {}"'
            os.system(cmd.format(os.path.join(run_path, 'hdf5_{}-to-{}.out'.format(target, start)), protein, target, start))
            # print(cmd.format(os.path.join(run_path, 'hdf5_{}-to-{}.out'.format(target, start)), protein, target, start))
        print(len(unfinished_pairs))


    if args.task == 'group':
        protein = args.protein
        target = args.target
        start = args.start
        print(protein, target, start)
        target_dir = os.path.join(args.datapath, target)
        start_dir = os.path.join(args.datapath, start)

        proteins = []
        pockets = []
        pdb_codes = []

        label_df = pd.read_csv(os.path.join(args.label_root, '{}-to-{}.csv'.format(target, start)))
        cif_files = fi.find_files(target_dir, 'mmcif')

        protein_file = '{}_protein.mmcif'.format(start)
        protein_df = dt.bp_to_df(dt.read_any(os.path.join(start_dir, protein_file)))

        big_sdf = os.path.join(args.out_dir, '{}-to-{}.sdf'.format(target, start))
        with open(big_sdf, 'w') as lig_fp:
            for f in tqdm(cif_files, desc='reading structures'):
                if '_pocket' in f:
                    id_code = f.split('_pocket')[-1].split('.mmcif')[0]
                    pdb_code = '{}_lig{}'.format(target, id_code)
                    try:
                        df = dt.bp_to_df(dt.read_any(os.path.join(target_dir, f)))
                        pockets.append(df)
                        pdb_codes.append(pdb_code)
                        proteins.append(protein_df)
                        with open(os.path.join(target_dir, '{}_ligand{}.sdf'.format(target, id_code))) as lig_f:
                            for lig_line in lig_f:
                                lig_fp.write(lig_line)
                    except Exception as e:
                        label_df = label_df[label_df['target'] != pdb_code]

        protein_df = pd.concat(proteins)
        pocket_df = pd.concat(pockets)
        pdb_codes = pd.DataFrame({'pdb': pdb_codes})
        label_df.drop(['protein', 'start'], axis=1)

        protein_df.to_csv(os.path.join(args.out_dir, '{}-to-{}_protein.csv'.format(target, start)))
        pocket_df.to_csv(os.path.join(args.out_dir, '{}-to-{}_pocket.csv'.format(target, start)))
        pdb_codes.to_csv(os.path.join(args.out_dir, '{}-to-{}_pdb_codes.csv'.format(target, start)))
        label_df.to_csv(os.path.join(args.out_dir, '{}-to-{}_labels.csv'.format(target, start)))

    if args.task == 'check':
        print('Missing:', len(unfinished_pairs), '/', len(pairs))
        # print(unfinished_pairs)

    if args.task == 'combine_prot':
        all_prot_df = []
        for protein, target, start in tqdm(pairs, desc='pairs in protein index file'):
            all_prot_df.append(pd.read_csv(os.path.join(args.out_dir, '{}-to-{}_protein.csv'.format(target, start))))
        protein_df = pd.concat(all_prot_df)
        protein_df.to_csv(os.path.join(args.root, 'proteins.csv'))

    if args.task == 'combine_pocket':
        all_pocket_df = []
        for protein, target, start in tqdm(pairs, desc='pairs in protein index file'):
            all_pocket_df.append(pd.read_csv(os.path.join(args.out_dir, '{}-to-{}_pocket.csv'.format(target, start))))
        pocket_df = pd.concat(all_pocket_df)
        pocket_df.to_csv(os.path.join(args.root, 'pockets.csv'))

    if args.task == 'combine_pdb':
        all_pdb_df = []
        for protein, target, start in tqdm(pairs, desc='pairs in protein index file'):
            all_pdb_df.append(pd.read_csv(os.path.join(args.out_dir, '{}-to-{}_pdb_codes.csv'.format(target, start))))
        pdb_codes = pd.concat(all_pdb_df)
        pdb_codes.to_csv(os.path.join(args.root, 'pdbs.csv'))

    if args.task == 'combine_label':
        all_labels_df = []
        for protein, target, start in tqdm(pairs, desc='pairs in protein index file'):
            all_labels_df.append(
                pd.read_csv(os.path.join(args.out_dir, '{}-to-{}_labels.csv'.format(target, start))))
        label_df = pd.concat(all_labels_df)
        label_df.to_csv(os.path.join(args.root, 'labels.csv'))

    if args.task == 'combine_ligand':
        all_lig_df = []
        for protein, target, start in tqdm(pairs, desc='pairs in protein index file'):
            all_lig_df.append(PandasTools.LoadSDF(os.path.join(args.out_dir, '{}-to-{}.sdf'.format(target, start)), molColName='Mol'))
        lig_df = pd.concat(all_lig_df)
        lig_df.to_csv(os.path.join(args.root, 'ligs.csv'))

    if args.task == 'combine_all':
        cmd = 'sbatch -p owners -t 5:00:00 -o {} --mem-per-cpu MaxMemPerCPU --wrap="' \
              '/home/groups/rondror/software/sidhikab/miniconda/envs/geometric/bin/python create_hdf5.py {} ' \
              '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/processed ' \
              '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/labels ' \
              '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/hdf5 ' \
              '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt ' \
              '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d"'
        # os.system(cmd.format(os.path.join(run_path, 'combine_prot.out'), 'combine_prot'))
        # os.system(cmd.format(os.path.join(run_path, 'combine_pocket.out'), 'combine_pocket'))
        # os.system(cmd.format(os.path.join(run_path, 'combine_pdb.out'), 'combine_pdb'))
        # os.system(cmd.format(os.path.join(run_path, 'combine_label.out'), 'combine_label'))
        os.system(cmd.format(os.path.join(run_path, 'combine_ligand.out'), 'combine_ligand'))

    if args.task == 'create_hdf5':
        hdf_file = os.path.join(args.root, 'pdbbind_refined.hdf5')

        protein_df = pd.read_csv(os.path.join(args.root, 'proteins.csv'))
        pocket_df = pd.read_csv(os.path.join(args.root, 'pockets.csv'))
        pdb_codes = pd.read_csv(os.path.join(args.root, 'pdbs.csv'))

        protein_df.to_hdf(hdf_file, 'proteins')
        pocket_df.to_hdf(hdf_file, 'pockets')
        pdb_codes.to_hdf(hdf_file, 'pdb_codes')

        print('converting ligands...')
        lig_df = pd.read_csv(os.path.join(args.root, 'ligs.csv'))
        lig_df.index = pdb_codes
        lig_df.to_hdf(hdf_file, 'ligands')

        print('converting labels...')
        label_df = pd.read_csv(os.path.join(args.root, 'labels.csv'))
        label_df = label_df.set_index('target').reindex(pdb_codes)
        label_df.to_hdf(hdf_file, 'labels')


if __name__ == "__main__":
    main()

