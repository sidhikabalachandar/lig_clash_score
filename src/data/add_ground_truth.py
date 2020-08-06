"""
The purpose of this code is to set the train, val, and test data sets
It can be run on sherlock using
/home/groups/rondror/software/sidhikab/miniconda/envs/geometric/bin/python add_ground_truth.py create /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt
/home/groups/rondror/software/sidhikab/miniconda/envs/geometric/bin/python add_ground_truth.py check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt --type <type>
"""

import argparse
import os

save_root = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw'
run_path = '/home/users/sidhikab/flexibility_project/atom3d/src/data/run'
data_root = '/oak/stanford/groups/rondror/projects/ligand-docking/pdbbind_2019/data'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either create or check')
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('--type', type=str, default='none', help='either prot, lig, or pv')
    args = parser.parse_args()

    if args.task == 'create':
        process = []
        with open(args.docked_prot_file) as fp:
            for line in fp:
                if line[0] == '#': continue
                protein, target, start = line.strip().split()
                protein_path = os.path.join(save_root, protein)
                pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
                if not os.path.exists(protein_path):
                    os.mkdir(protein_path)
                if not os.path.exists(pair_path):
                    os.mkdir(pair_path)
                process.append((protein, target, start))

        grouped_files = []
        n = 25

        for i in range(0, len(process), n):
            grouped_files += [process[i: i + n]]

        if not os.path.exists(run_path):
            os.mkdir(run_path)

        for i, group in enumerate(grouped_files):
            print('converting to pdb', i)

            with open(os.path.join(run_path, 'structure{}_in.sh'.format(i)), 'w') as f:
                for protein, target, start in group:
                    protein_path = os.path.join(save_root, protein)
                    pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
                    struct_root = os.path.join(data_root, '{}/structures/aligned'.format(protein))
                    dock_root = os.path.join(data_root, '{}/docking/sp_es4/{}-to-{}'.format(protein, target, start))
                    f.write('#!/bin/bash\n')
                    if not os.path.exists('{}/{}_prot.mae'.format(pair_path, start)):
                        f.write('cp {}/{}_prot.mae {}/{}_prot.mae\n'.format(struct_root, start, pair_path, start))
                    if not os.path.exists('{}/{}_prot.mae'.format(pair_path, target)):
                        f.write('cp {}/{}_prot.mae {}/{}_prot.mae\n'.format(struct_root, target, pair_path, target))
                    if not os.path.exists('{}/{}_lig.mae'.format(pair_path, start)):
                        f.write('cp {}/{}_lig.mae {}/{}_lig.mae\n'.format(struct_root, start, pair_path, start))
                    if not os.path.exists('{}/{}_lig0.mae'.format(pair_path, target)):
                        f.write('cp {}/{}_lig.mae {}/{}_lig0.mae\n'.format(struct_root, target, pair_path, target))
                    if not os.path.exists('{}/{}-to-{}_pv.maegz'.format(pair_path, target, start)):
                        f.write('cp {}/{}-to-{}_pv.maegz {}/{}-to-{}_pv.maegz\n'.format(dock_root, target, start, pair_path, target, start))

            os.chdir(run_path)
            os.system('sbatch -p owners -t 02:00:00 -o structure{}.out structure{}_in.sh'.format(i, i))
            # print('sbatch -p owners -t 02:00:00 -o structure{}.out structure{}_in.sh'.format(i, i))

    elif args.task == 'check':
        process = []
        counter = 0
        with open(args.docked_prot_file) as fp:
            for line in fp:
                if line[0] == '#': continue
                counter += 1
                protein, target, start = line.strip().split()
                protein_path = os.path.join(save_root, protein)
                pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
                if not os.path.exists(protein_path):
                    os.mkdir(protein_path)
                if not os.path.exists(pair_path):
                    os.mkdir(pair_path)
                if args.type == 'prot':
                    if not os.path.exists('{}/{}_prot.mae'.format(pair_path, start)):
                        process.append('{}/{}_prot.mae'.format(pair_path, start))
                    if not os.path.exists('{}/{}_prot.mae'.format(pair_path, target)):
                        process.append('{}/{}_prot.mae'.format(pair_path, target))
                elif args.type == 'lig':
                    if not os.path.exists('{}/{}_lig.mae'.format(pair_path, start)):
                        process.append('{}/{}_lig.mae'.format(pair_path, start))
                    if not os.path.exists('{}/{}_lig0.mae'.format(pair_path, target)):
                        process.append('{}/{}_lig0.mae'.format(pair_path, target))
                elif args.type == 'pv':
                    if not os.path.exists('{}/{}-to-{}_pv.maegz'.format(pair_path, target, start)):
                        process.append((protein, target, start))

        print('Missing', len(process), '/', counter)
        print(process)

if __name__=="__main__":
    main()
