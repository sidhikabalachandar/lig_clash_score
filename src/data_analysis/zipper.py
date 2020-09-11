"""
The purpose of this code is to create the .zip structure files

It can be run on sherlock using
$ ml load chemistry
$ ml load schrodinger
$ $SCHRODINGER/run python3 zipper.py run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
$ $SCHRODINGER/run python3 zipper.py check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
$ $SCHRODINGER/run python3 zipper.py update /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw --new_prot_file /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random2.txt
$ $SCHRODINGER/run python3 zipper.py remove_in_log /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
"""

import os
from schrodinger.structutils.transform import get_centroid
import schrodinger.structure as structure
import argparse
from tqdm import tqdm

N = 1
MAX_POSES = 100
MAX_DECOYS = 10

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either run, check, remove_pv, or MAPK14')
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('--new_prot_file', type=str, default=os.path.join(os.getcwd(), 'index.txt'),
                        help='for update task, name of new prot file')
    args = parser.parse_args()

    if args.task == 'run':
        # process = get_prots(args.docked_prot_file)
        process = [('P04746', '3old', '1xd0')]
        grouped_files = group_files(N, process)

        for i, group in enumerate(grouped_files):
            with open(os.path.join(args.run_path, 'grid{}_in.sh'.format(i)), 'w') as f:
                for protein, target, start in group:
                    protein_path = os.path.join(args.raw_root, protein)
                    pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
                    pose_path = os.path.join(pair_path, 'ligand_poses')

                    with open(os.path.join(pair_path, '{}-to-{}.in'.format(target, start)), 'w') as f_in:
                        c = get_centroid(list(structure.StructureReader(os.path.join(pose_path,
                                                                                     '{}_lig0.mae'.format(target))))[0])
                        x, y, z = c[:3]

                        f_in.write('GRID_CENTER {},{},{}\n'.format(x, y, z))
                        f_in.write('GRIDFILE {}-to-{}.zip\n'.format(target, start))
                        f_in.write('INNERBOX 15,15,15\n')
                        f_in.write('OUTERBOX 30,30,30\n')
                        f_in.write('RECEP_FILE {}\n'.format(os.path.join(pair_path, '{}_prot.mae'.format(start))))

                        f.write('#!/bin/bash\n')
                        f.write('cd {}\n'.format(pair_path))
                        f.write('$SCHRODINGER/glide -WAIT {}-to-{}.in\n'.format(target, start))
                        f.write('rm {}-to-{}.in'.format(target, start))
                        f.write('rm {}-to-{}.log'.format(target, start))

            os.chdir(args.run_path)
            os.system('sbatch -p owners -t 00:30:00 -o grid{}.out grid{}_in.sh'.format(i, i))

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

                # check basic files
                if not os.path.exists('{}/{}-to-{}.zip'.format(pair_path, target, start)):
                    process.append((protein, target, start))

        print('Missing', len(process), '/', num_pairs)
        print(process)

    if args.task == 'update':
        text = []
        with open(args.docked_prot_file) as fp:
            for line in tqdm(fp, desc='files'):
                if line[0] == '#': continue
                protein, target, start = line.strip().split()
                protein_path = os.path.join(args.raw_root, protein)
                pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
                if os.path.exists('{}/{}-to-{}.zip'.format(pair_path, target, start)):
                    text.append(line)

        file = open(args.new_prot_file, "w")
        file.writelines(text)
        file.close()

if __name__ == '__main__':
    main()