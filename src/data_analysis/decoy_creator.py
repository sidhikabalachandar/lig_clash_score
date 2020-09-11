"""
The purpose of this code is to create the translated and rotated ligand poses for each of the top glide poses

It can be run on sherlock using
ml load chemistry
ml load schrodinger
$ $SCHRODINGER/run python3 decoy_creator.py all /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
$ $SCHRODINGER/run python3 decoy_creator.py group /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw --index <index>
$ $SCHRODINGER/run python3 decoy_creator.py check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
$ $SCHRODINGER/run python3 decoy_creator.py all_dist_check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
$ $SCHRODINGER/run python3 decoy_creator.py group_dist_check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw --index <index>
$ $SCHRODINGER/run python3 decoy_creator.py check_dist_check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
$ $SCHRODINGER/run python3 decoy_creator.py all_name_check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
$ $SCHRODINGER/run python3 decoy_creator.py group_name_check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw --index <index>
$ $SCHRODINGER/run python3 decoy_creator.py check_name_check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
$ $SCHRODINGER/run python3 decoy_creator.py MAPK14
"""

import os
import schrodinger.structure as structure
import schrodinger.structutils.transform as transform
from schrodinger.structutils.transform import get_centroid
import numpy as np
import argparse
from tqdm import tqdm
import statistics
import pickle

MAX_POSES = 100
MAX_DECOYS = 10
MEAN_TRANSLATION = 0
STDEV_TRANSLATION = 1
MIN_ANGLE = - np.pi / 6
MAX_ANGLE = np.pi / 6
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

def cartesian_vector(i):
    """
    Generates a random 3D unit vector (direction) with a uniform spherical distribution
    Algo from http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
    :return:
    """
    if i == 0:
        return 1, 0, 0
    elif i == 1:
        return -1, 0, 0
    elif i == 2:
        return 0, 1, 0
    elif i == 3:
        return 0, -1, 0
    elif i == 4:
        return 0, 0, 1
    elif i == 5:
        return 0, 0, -1
    else:
        print('Bad input')


def modify_file(path, name):
    reading_file = open(path, "r")
    file_name = path.split('/')[-1]

    new_file_content = ""
    for line in reading_file:
        if line.strip() == name:
            new_line = line.replace(name, file_name)
        else:
            new_line = line
        new_file_content += new_line
    reading_file.close()

    writing_file = open(path, "w")
    writing_file.write(new_file_content)
    writing_file.close()

def create_decoys(lig_file):
    """
    creates MAX_DECOYS number of translated/rotated decoys
    :param lig_file: (string) file of glide ligand pose that will be translated/rotated
    :return:
    """
    code = lig_file.split('/')[-1].split('_')[-1]
    if code == 'lig0.mae':
        modify_file(lig_file, '_pro_ligand')
    else:
        modify_file(lig_file, '_ligand')
    for i in range(MAX_DECOYS):
        s = list(structure.StructureReader(lig_file))[0]

        #translation
        x, y, z = random_three_vector()
        dist = np.random.normal(MEAN_TRANSLATION, STDEV_TRANSLATION)
        transform.translate_structure(s, x * dist, y * dist, z * dist)

        #rotation
        x_angle = np.random.uniform(MIN_ANGLE, MAX_ANGLE)
        y_angle = np.random.uniform(MIN_ANGLE, MAX_ANGLE)
        z_angle = np.random.uniform(MIN_ANGLE, MAX_ANGLE)
        rot_center = list(get_centroid(s))
        transform.rotate_structure(s, x_angle, y_angle, z_angle, rot_center)

        decoy_file = lig_file[:-4] + chr(ord('a')+i) + '.mae'
        with structure.StructureWriter(decoy_file) as decoy:
            decoy.append(s)
        if code == 'lig0.mae':
            modify_file(decoy_file, lig_file.split('/')[-1])
        else:
            modify_file(decoy_file, lig_file.split('/')[-1])

def create_cartesian_decoys(lig_file):
    """
    creates MAX_DECOYS number of translated/rotated decoys
    :param lig_file: (string) file of glide ligand pose that will be translated/rotated
    :return:
    """
    code = lig_file.split('/')[-1].split('_')[-1]
    if code == 'lig0.mae':
        modify_file(lig_file, '_pro_ligand')
    else:
        modify_file(lig_file, '_ligand')
    for i in range(6):
        s = list(structure.StructureReader(lig_file))[0]

        #translation
        x, y, z = cartesian_vector(i)
        dist = np.random.normal(MEAN_TRANSLATION, STDEV_TRANSLATION)
        transform.translate_structure(s, x * dist, y * dist, z * dist)

        decoy_file = lig_file[:-4] + chr(ord('a')+i) + '.mae'
        with structure.StructureWriter(decoy_file) as decoy:
            decoy.append(s)
        if code == 'lig0.mae':
            modify_file(decoy_file, lig_file.split('/')[-1])
        else:
            modify_file(decoy_file, lig_file.split('/')[-1])

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
    parser.add_argument('--dist_dir', type=str, default=os.path.join(os.getcwd(), 'dists'),
                        help='for all_dist_check and group_dist_check task, directiory to place distances')
    parser.add_argument('--name_dir', type=str, default=os.path.join(os.getcwd(), 'names'),
                        help='for all_name_check and group_name_check task, directiory to place unfinished protein, '
                             'target, start groups')
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
            pose_path = os.path.join(pair_path, 'cartesian_ligand_poses')
            pv_file = os.path.join(pair_path, '{}-to-{}_glide_pv.maegz'.format(target, start))
            num_poses = len(list(structure.StructureReader(pv_file)))
            print(num_poses)

            for i in range(num_poses):
                if i == MAX_POSES:
                    break
                lig_file = os.path.join(pose_path, '{}_lig{}.mae'.format(target, i))
                create_cartesian_decoys(lig_file)
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

    if args.task == 'all_dist_check':
        # if not os.path.exists(args.dist_dir):
        #     os.mkdir(args.dist_dir)
        #
        # process = get_prots(args.docked_prot_file)
        # grouped_files = group_files(N, process)

        groups = [31, 32, 151, 176, 186, 187, 189, 194, 195, 198, 225, 226, 322, 332, 333, 341, 343, 452, 453, 460, 487,
                  495]

        if not os.path.exists(args.run_path):
            os.mkdir(args.run_path)

        # for i, group in enumerate(grouped_files):
        for i in groups:
            cmd = 'sbatch -p owners -t 0:20:00 -o {} --wrap="$SCHRODINGER/run python3 decoy_creator.py group_dist_check {} {} {} ' \
                  '--index {}"'
            os.system(cmd.format(os.path.join(args.run_path, 'decoy{}.out'.format(i)), args.docked_prot_file,
                                 args.run_path, args.raw_root, i))

    if args.task == 'group_dist_check':
        if not os.path.exists(args.dist_dir):
            os.mkdir(args.dist_dir)

        process = get_prots(args.docked_prot_file)
        grouped_files = group_files(N, process)
        save = []

        for protein, target, start in grouped_files[args.index]:
            protein_path = os.path.join(args.raw_root, protein)
            pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
            pose_path = os.path.join(pair_path, 'ligand_poses')
            pv_file = os.path.join(pair_path, '{}-to-{}_pv.maegz'.format(target, start))
            num_poses = len(list(structure.StructureReader(pv_file)))
            means = []

            for i in range(num_poses):
                if i == MAX_POSES:
                    break
                lig_file = os.path.join(pose_path, '{}_lig{}.mae'.format(target, i))
                s = list(structure.StructureReader(lig_file))[0]
                c = get_centroid(s)
                dists = []

                for j in range(MAX_DECOYS):
                    decoy_file = lig_file[:-4] + chr(ord('a')+j) + '.mae'
                    decoy = list(structure.StructureReader(decoy_file))[0]
                    dists.append(transform.get_vector_magnitude(c - get_centroid(decoy)))

                means.append(statistics.mean(dists))

            save.append(statistics.mean(means))

        outfile = open(os.path.join(args.dist_dir, '{}.pkl'.format(args.index)), 'wb')
        pickle.dump(save, outfile)
        print(save)

    if args.task == 'check_dist_check':
        process = get_prots(args.docked_prot_file)
        grouped_files = group_files(N, process)

        if len(os.listdir(args.dist_dir)) != len(grouped_files):
            print('Not all files created')
        else:
            print('All files created')

        errors = []
        for i in range(len(grouped_files)):
            infile = open(os.path.join(args.dist_dir, '{}.pkl'.format(i)), 'rb')
            vals = pickle.load(infile)
            infile.close()

            for j in vals:
                if j > 2 or j < -1:
                    print(vals)
                    errors.append(i)
                    break

        print('Potential errors', len(errors), '/', len(grouped_files))
        print(errors)

    if args.task == 'all_name_check':
        if not os.path.exists(args.name_dir):
            os.mkdir(args.name_dir)

        process = get_prots(args.docked_prot_file)
        grouped_files = group_files(N, process)

        if not os.path.exists(args.run_path):
            os.mkdir(args.run_path)

        for i, group in enumerate(grouped_files):
            cmd = 'sbatch -p owners -t 0:20:00 -o {} --wrap="$SCHRODINGER/run python3 decoy_creator.py group_name_check {} {} {} ' \
                  '--index {}"'
            os.system(cmd.format(os.path.join(args.run_path, 'name{}.out'.format(i)), args.docked_prot_file,
                                 args.run_path, args.raw_root, i))

    if args.task == 'group_name_check':
        if not os.path.exists(args.name_dir):
            os.mkdir(args.name_dir)

        process = get_prots(args.docked_prot_file)
        grouped_files = group_files(N, process)
        unfinished = []

        for protein, target, start in grouped_files[args.index]:
            protein_path = os.path.join(args.raw_root, protein)
            pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
            pose_path = os.path.join(pair_path, 'ligand_poses')
            pv_file = os.path.join(pair_path, '{}-to-{}_glide_pv.maegz'.format(target, start))
            num_poses = len(list(structure.StructureReader(pv_file)))

            for i in range(num_poses):
                if i == MAX_POSES:
                    break
                lig_file = os.path.join(pose_path, '{}_lig{}.mae'.format(target, i))
                found = False
                with open(lig_file, "r") as f:
                    file_name = lig_file.split('/')[-1]
                    for line in f:
                        if line.strip() == file_name:
                            found = True
                if not found:
                    print(lig_file)
                    unfinished.append((protein, target, start))
                    break
                else:
                    for j in range(MAX_DECOYS):
                        decoy_file = lig_file[:-4] + chr(ord('a') + j) + '.mae'
                        found = False
                        with open(decoy_file, "r") as f:
                            file_name = decoy_file.split('/')[-1]
                            for line in f:
                                if line.strip() == file_name:
                                    found = True
                        if not found:
                            print(decoy_file)
                            unfinished.append((protein, target, start))
                            break
                if not found:
                    break
            break

        # outfile = open(os.path.join(args.name_dir, '{}.pkl'.format(args.index)), 'wb')
        # pickle.dump(unfinished, outfile)
        print(unfinished)

    if args.task == 'check_name_check':
        process = get_prots(args.docked_prot_file)
        grouped_files = group_files(N, process)

        if len(os.listdir(args.name_dir)) != len(grouped_files):
            print('Not all files created')
        else:
            print('All files created')

        errors = []
        for i in range(len(grouped_files)):
            infile = open(os.path.join(args.name_dir, '{}.pkl'.format(i)), 'rb')
            unfinished = pickle.load(infile)
            infile.close()
            errors.extend(unfinished)

        print('Errors', len(errors), '/', len(process))
        print(errors)

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