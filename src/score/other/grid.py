"""
The purpose of this code is to create the .zip structure files

It can be run on sherlock using
$ ml load chemistry
$ ml load schrodinger
$ $SCHRODINGER/run python3 grid.py run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
$ $SCHRODINGER/run python3 grid.py check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw

$ $SCHRODINGER/run python3 grid.py run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits/combined_index_large.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw

$ $SCHRODINGER/run python3 grid.py update /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw --new_prot_file /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random2.txt

"""

import os
from schrodinger.structutils.transform import get_centroid
import schrodinger.structure as structure
import argparse
from tqdm import tqdm


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

def run(grouped_files, run_path, raw_root, decoy_type):
    """
    creates grid for each protein, target, start
    :param grouped_files: (list) list of protein, target, start groups
    :param run_path: (string) directory where script and output files will be written
    :param raw_root: (string) directory where raw data will be placed
    :return:
    """
    for i, group in enumerate(grouped_files):
        with open(os.path.join(run_path, 'grid{}_in.sh'.format(i)), 'w') as f:
            print(os.path.join(run_path, 'grid{}_in.sh'.format(i)))
            for protein, target, start in group:
                pair = '{}-to-{}'.format(target, start)
                target_pair = '{}-to-{}'.format(target, target)
                protein_path = os.path.join(raw_root, protein)
                pair_path = os.path.join(protein_path, pair)
                pose_path = os.path.join(pair_path, decoy_type)
                print(pair_path)

                # create in file for grid
                if not os.path.exists(os.path.join(pair_path, '{}.zip'.format(pair))):
                    with open(os.path.join(pair_path, '{}.in'.format(pair)), 'w') as f_in:
                        c = get_centroid(list(structure.StructureReader(os.path.join(pose_path,
                                                                                     '{}_lig0.mae'.format(target))))[0])
                        x, y, z = c[:3]

                        f_in.write('GRID_CENTER {},{},{}\n'.format(x, y, z))
                        f_in.write('GRIDFILE {}.zip\n'.format(pair))
                        f_in.write('INNERBOX 15,15,15\n')
                        f_in.write('OUTERBOX 30,30,30\n')
                        f_in.write('RECEP_FILE {}\n'.format(os.path.join(pair_path, '{}_prot.mae'.format(start))))
                        # create grid commands
                        f.write('#!/bin/bash\n')
                        f.write('cd {}\n'.format(pair_path))
                        f.write('$SCHRODINGER/glide -WAIT {}.in\n'.format(pair))
                        f.write('rm {}/{}.in\n'.format(pair_path, pair))
                        f.write('rm {}/{}.log\n'.format(pair_path, pair))

                if not os.path.exists(os.path.join(pair_path, '{}.zip'.format(target_pair))):
                    print('hi')
                    with open(os.path.join(pair_path, '{}.in'.format(target_pair)), 'w') as f_in:
                        c = get_centroid(list(structure.StructureReader(os.path.join(pose_path,
                                                                                     '{}_lig0.mae'.format(target))))[0])
                        x, y, z = c[:3]

                        f_in.write('GRID_CENTER {},{},{}\n'.format(x, y, z))
                        f_in.write('GRIDFILE {}.zip\n'.format(target_pair))
                        f_in.write('INNERBOX 15,15,15\n')
                        f_in.write('OUTERBOX 30,30,30\n')
                        f_in.write('RECEP_FILE {}\n'.format(os.path.join(pair_path, '{}_prot.mae'.format(target))))
                        # create grid commands
                        f.write('#!/bin/bash\n')
                        f.write('cd {}\n'.format(pair_path))
                        f.write('$SCHRODINGER/glide -WAIT {}.in\n'.format(target_pair))
                        f.write('rm {}/{}.in\n'.format(pair_path, target_pair))
                        f.write('rm {}/{}.log\n'.format(pair_path, target_pair))
                break

        os.chdir(run_path)
        os.system('sbatch -p rondror -t 02:00:00 -o grid{}.out grid{}_in.sh'.format(i, i))
        # print('sbatch -p owners -t 02:00:00 -o grid{}.out grid{}_in.sh'.format(i, i))
        break

def check(docked_prot_file, raw_root):
    """
    check if all grids created
    :param docked_prot_file: (string) file listing proteins to process
    :param raw_root: (string) directory where raw data will be placed
    :return:
    """
    process = []
    num_pairs = 0
    with open(docked_prot_file) as fp:
        for line in tqdm(fp, desc='going through protein, target, start groups'):
            if line[0] == '#': continue
            protein, target, start = line.strip().split()
            num_pairs += 1
            protein_path = os.path.join(raw_root, protein)
            pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))

            if not os.path.exists('{}/{}-to-{}.zip'.format(pair_path, target, start)):
                process.append((protein, target, start))
                continue

    print('Missing', len(process), '/', num_pairs)
    print(process)

def update(docked_prot_file, raw_root, new_prot_file):
    """
    update index by removing protein, target, start that could not create grids
    :param docked_prot_file: (string) file listing proteins to process
    :param raw_root: (string) directory where raw data will be placed
    :param new_prot_file: (string) name of new prot file
    :return:
    """
    text = []
    with open(docked_prot_file) as fp:
        for line in tqdm(fp, desc='files'):
            if line[0] == '#': continue
            protein, target, start = line.strip().split()
            protein_path = os.path.join(raw_root, protein)
            pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
            if os.path.exists('{}/{}-to-{}.zip'.format(pair_path, target, start)):
                text.append(line)

    file = open(new_prot_file, "w")
    file.writelines(text)
    file.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either run, check, or update')
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('--new_prot_file', type=str, default=os.path.join(os.getcwd(), 'index.txt'),
                        help='for update task, name of new prot file')
    parser.add_argument('--n', type=int, default=3, help='number of protein, target, start groups processed in '
                                                         'group task')
    parser.add_argument('--decoy_type', type=str, default='ligand_poses', help='either cartesian_poses, ligand_poses, '
                                                                               'or conformer_poses')
    args = parser.parse_args()

    if not os.path.exists(args.run_path):
        os.mkdir(args.run_path)

    if args.task == 'run':
        process = get_prots(args.docked_prot_file)
        grouped_files = group_files(args.n, process)
        run(grouped_files, args.run_path, args.raw_root, args.decoy_type)

    if args.task == 'check':
        check(args.docked_prot_file, args.raw_root)

    if args.task == 'update':
        update(args.docked_prot_file, args.raw_root, args.new_prot_file)

if __name__ == '__main__':
    main()