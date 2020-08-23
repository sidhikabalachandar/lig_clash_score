"""
The purpose of this code is to create the translated and rotated ligand poses for each of the top glide poses

It can be run on sherlock using
ml load chemistry
ml load schrodinger
$ $SCHRODINGER/run python3 decoy_creator.py all /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
$ $SCHRODINGER/run python3 decoy_creator.py group /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw --index <index>
$ $SCHRODINGER/run python3 decoy_creator.py check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
$ $SCHRODINGER/run python3 decoy_creator.py MAPK14
"""

import os
import schrodinger.structure as structure
import schrodinger.structutils.transform as transform
import numpy as np
import argparse
from tqdm import tqdm

MAX_POSES = 100
MAX_DECOYS = 10
MEAN_TRANSLATION = 6
STDEV_TRANSLATION = 4
N = 3

def random_three_vector():
    """
    Generates a random 3D unit vector (direction) with a uniform spherical distribution
    Algo from http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
    :return:
    """
    phi = np.random.uniform(0,np.pi*2)
    costheta = np.random.uniform(-1,1)

    theta = np.arccos(costheta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z

def create_decoys(lig_file):
    """
    creates MAX_DECOYS number of translated/rotated decoys
    :param lig_file: (string) file of glide ligand pose that will be translated/rotated
    :return:
    """
    s = list(structure.StructureReader(lig_file))[0]
    for i in range(MAX_DECOYS):
        #translation
        x, y, z = random_three_vector()
        dist = np.random.normal(MEAN_TRANSLATION, STDEV_TRANSLATION)
        coords = s.getXYZ()
        coords += np.array([x * dist, y * dist, z * dist])
        s.setXYZ(coords)

        #rotation
        x_angle = np.random.uniform(0, np.pi * 2)
        y_angle = np.random.uniform(0, np.pi * 2)
        z_angle = np.random.uniform(0, np.pi * 2)
        rot_center = list(np.mean(coords, axis=0))
        transform.rotate_structure(s, x_angle, y_angle, z_angle, rot_center)

        with structure.StructureWriter(lig_file[:-4] + chr(ord('a')+i) + '.mae') as decoy:
            decoy.append(s)

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
    parser.add_argument('task', type=str, help='either all, group, check, or MAPK14')
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('--index', type=int, default=-1, help='for group task, group number')
    args = parser.parse_args()

    if args.task == 'all':
        process = get_prots(args.docked_prot_file)
        grouped_files = group_files(N, process)

        if not os.path.exists(args.run_path):
            os.mkdir(args.run_path)

        for i, group in enumerate(grouped_files):
            cmd = 'sbatch -p owners -t 1:00:00 -o {} --wrap="$SCHRODINGER/run python3 decoy_creator.py group {} {} {} ' \
                  '--index {}"'
            os.system(cmd.format(os.path.join(args.run_path, 'decoy{}.out'.format(i)), args.docked_prot_file,
                                 args.run_path, args.raw_root, i))

    if args.task == 'group':
        process = get_prots(args.docked_prot_file)
        grouped_files = group_files(N, process)

        for protein, target, start in grouped_files[args.index]:
            protein_path = os.path.join(args.raw_root, protein)
            pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
            pose_path = os.path.join(pair_path, 'ligand_poses')
            pv_file = os.path.join(pair_path, '{}-to-{}_pv.maegz'.format(target, start))
            num_poses = len(list(structure.StructureReader(pv_file)))
            for i in range(0, num_poses):
                if i == MAX_POSES:
                    break
                lig_file = os.path.join(pose_path, '{}_lig{}.mae'.format(target, i))
                create_decoys(lig_file)
                break

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
                pose_path = os.path.join(pair_path, 'ligand_poses')
                pv_file = os.path.join(pair_path, '{}-to-{}_pv.maegz'.format(target, start))

                # num_poses = min(MAX_POSES, len(list(structure.StructureReader(pv_file))))
                num_poses = 0
                for i in range(MAX_DECOYS):
                    if not os.path.join(pose_path, '{}_lig{}.mae'.format(target, str(num_poses) + chr(ord('a')+i))):
                        process.append((protein, target, start))
                        print(os.path.join(pose_path, '{}_lig{}.mae'.format(target, str(num_poses) + chr(ord('a')+i))))
                        break

        print('Missing', len(process), '/', num_pairs)
        print(process)

    if args.task == 'MAPK14':
        protein = 'MAPK14'
        ligs = ['3D83', '4F9Y']
        for target in ligs:
            for start in ligs:
                if target != start:
                    file = os.path.join(args.raw_root,
                                        '{}/{}-to-{}/{}-to-{}_pv.maegz'.format(protein, target, start, target, start))
                    num_poses = len(list(structure.StructureReader(file)))
                    for i in range(num_poses):
                        if i == 101:
                            break
                        lig_file = '{}/{}/{}-to-{}/{}_lig{}.mae'.format(args.raw_root, protein, target, start, target, i)
                        create_decoys(lig_file)

if __name__=="__main__":
    main()