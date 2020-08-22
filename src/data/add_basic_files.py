"""
The purpose of this code is to create the raw directory folder and include the following files
starting protein receptor
starting ligand
target ligand
glide pose viewer file

It can be run on sherlock using
/home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python add_basic_files.py create /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /oak/stanford/groups/rondror/projects/ligand-docking/pdbbind_2019/data
/home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python add_basic_files.py check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /oak/stanford/groups/rondror/projects/ligand-docking/pdbbind_2019/data --type prot
/home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python add_basic_files.py check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /oak/stanford/groups/rondror/projects/ligand-docking/pdbbind_2019/data --type lig
/home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python add_basic_files.py check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /oak/stanford/groups/rondror/projects/ligand-docking/pdbbind_2019/data --type pv
/home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python add_basic_files.py MAPK14 /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/bpp_data/MAPK14/structures
"""

import argparse
import os

N = 25 #number of files in each group

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

    for i in range(0, len(process), n):
        grouped_files += [process[i: i + n]]

    return grouped_files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either create, check, or MAPK14')
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('data_root', type=str, help='pdbbind directory where raw data will be obtained')
    parser.add_argument('--type', type=str, default='none', help='for check task, either prot, lig, or pv')
    args = parser.parse_args()

    if args.task == 'create':
        process = get_prots(args.docked_prot_file)
        grouped_files = group_files(N, process)

        if not os.path.exists(args.run_path):
            os.mkdir(args.run_path)

        for i, group in enumerate(grouped_files):
            with open(os.path.join(args.run_path, 'structure{}_in.sh'.format(i)), 'w') as f:
                for protein, target, start in group:
                    dock_root = os.path.join(args.data_root,
                                             '{}/docking/sp_es4/{}-to-{}'.format(protein, target, start))
                    protein_path = os.path.join(args.raw_root, protein)
                    pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
                    struct_root = os.path.join(args.data_root, '{}/structures/aligned'.format(protein))
                    pose_path = os.path.join(pair_path, 'ligand_poses')

                    if not os.path.exists(args.raw_root):
                        os.mkdir(args.raw_root)
                    if not os.path.exists(protein_path):
                        os.mkdir(protein_path)
                    if not os.path.exists(pair_path):
                        os.mkdir(pair_path)
                    if not os.path.exists(pose_path):
                        os.mkdir(pose_path)

                    f.write('#!/bin/bash\n')
                    if not os.path.exists('{}/{}_prot.mae'.format(pair_path, start)):
                        f.write('cp {}/{}_prot.mae {}/{}_prot.mae\n'.format(struct_root, start, pair_path, start))
                    if not os.path.exists('{}/{}_lig.mae'.format(pair_path, start)):
                        f.write('cp {}/{}_lig.mae {}/{}_lig.mae\n'.format(struct_root, start, pair_path, start))
                    if not os.path.exists('{}/{}_lig0.mae'.format(pair_path, target)):
                        f.write('cp {}/{}_lig.mae {}/{}_lig0.mae\n'.format(struct_root, target, pose_path, target))
                    if not os.path.exists('{}/{}-to-{}_pv.maegz'.format(pair_path, target, start)):
                        f.write('cp {}/{}-to-{}_pv.maegz {}/{}-to-{}_pv.maegz\n'.format(dock_root, target, start, pair_path, target, start))

            os.chdir(args.run_path)
            os.system('sbatch -p owners -t 02:00:00 -o structure{}.out structure{}_in.sh'.format(i, i))
            # print('sbatch -p owners -t 02:00:00 -o structure{}.out structure{}_in.sh'.format(i, i))

    elif args.task == 'check':
        process = []
        num_pairs = 0
        with open(args.docked_prot_file) as fp:
            for line in fp:
                if line[0] == '#': continue
                protein, target, start = line.strip().split()
                num_pairs += 1
                protein_path = os.path.join(args.raw_root, protein)
                pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
                pose_path = os.path.join(pair_path, 'ligand_poses')

                if args.type == 'prot':
                    if not os.path.exists('{}/{}_prot.mae'.format(pair_path, start)):
                        process.append('{}/{}_prot.mae'.format(pair_path, start))
                elif args.type == 'lig':
                    if not os.path.exists('{}/{}_lig.mae'.format(pair_path, start)):
                        process.append('{}/{}_lig.mae'.format(pair_path, start))
                    if not os.path.exists('{}/{}_lig0.mae'.format(pose_path, target)):
                        process.append('{}/{}_lig0.mae'.format(pose_path, target))
                elif args.type == 'pv':
                    if not os.path.exists('{}/{}-to-{}_pv.maegz'.format(pair_path, target, start)):
                        process.append((protein, target, start))

        print('Missing', len(process), '/', num_pairs)
        print(process)

    elif args.task == 'MAPK14':
        protein = 'MAPK14'
        ligs = ['3D83', '4F9Y']
        dock_root = '/oak/stanford/groups/rondror/projects/combind/flexibility/MAPK14/MAPK14_regular/mut_rmsds/MAPK14'
        with open(os.path.join(args.run_path, 'structure_in.sh'), 'w') as f:
            for target in ligs:
                for start in ligs:
                    if target != start:
                        protein_path = os.path.join(args.raw_root, protein)
                        pair_path = os.path.join(protein_path, '{}-to-{}'.format(target.lower(), start.lower()))
                        struct_root = os.path.join(args.data_root, 'aligned_files')
                        lig_root = os.path.join(args.data_root, 'ligands')
                        pv_root = os.path.join(dock_root, '{}_to_{}'.format(target, start))

                        if not os.path.exists(pair_path):
                            os.mkdir(pair_path)

                        f.write('#!/bin/bash\n')
                        if not os.path.exists('{}/{}_prot.mae'.format(pair_path, start)):
                            f.write('cp {}/{}/{}_out.mae {}/{}_prot.mae\n'.format(struct_root, start, start, pair_path, start))
                        if not os.path.exists('{}/{}_lig.mae'.format(pair_path, start)):
                            f.write('cp {}/{}_lig.mae {}/{}_lig.mae\n'.format(lig_root, start, pair_path, start))
                        if not os.path.exists('{}/{}_lig0.mae'.format(pair_path, target)):
                            f.write('cp {}/{}_lig.mae {}/{}_lig0.mae\n'.format(lig_root, target, pair_path, target))
                        if not os.path.exists('{}/{}-to-{}_pv.maegz'.format(pair_path, target, start)):
                            f.write('cp {}/{}_to_{}_pv.maegz {}/{}-to-{}_pv.maegz\n'.format(pv_root, target, start, pair_path,
                                                                                            target, start))

        os.chdir(args.run_path)
        os.system('sbatch -p rondror -t 02:00:00 -o structure.out structure_in.sh')

if __name__=="__main__":
    main()
