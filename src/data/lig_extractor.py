"""
The purpose of this code is to unpack the pv file into one mae file for each ligand

It can be run on sherlock using
ml load chemistry
ml load schrodinger
$ $SCHRODINGER/run python3 lig_extractor.py all /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
$ $SCHRODINGER/run python3 lig_extractor.py group /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw --index <index>
$ $SCHRODINGER/run python3 lig_extractor.py check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
$ $SCHRODINGER/run python3 lig_extractor.py MAPK14
"""

import argparse
import os
import schrodinger.structure as structure

MAX_POSES = 100
N = 20

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
            cmd = 'sbatch -p owners -t 1:00:00 -o {} --wrap="$SCHRODINGER/run python3 lig_extractor.py group {} {} {} ' \
                  '--index {}"'
            os.system(cmd.format(os.path.join(args.run_path, 'lig{}.out'.format(i)), args.docked_prot_file,
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
            for i in range(1, num_poses):
                if i == MAX_POSES:
                    break
                with structure.StructureWriter(os.path.join(pose_path, '{}_lig{}.mae'.format(target, i))) as all:
                    all.append(list(structure.StructureReader(pv_file))[i])

    if args.task == 'check':
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
                pv_file = os.path.join(pair_path, '{}-to-{}_pv.maegz'.format(target, start))

                num_poses = min(MAX_POSES, len(list(structure.StructureReader(pv_file))))
                if not os.path.join(pose_path, '{}_lig{}.mae'.format(target, num_poses)):
                    process.append((protein, target, start))

        print('Missing', len(process), '/', num_pairs)
        print(process)

    if args.task == 'MAPK14':
        protein = 'MAPK14'
        ligs = ['3D83', '4F9Y']
        for target in ligs:
            for start in ligs:
                if target != start:
                    file = os.path.join(args.save_root,
                                       '{}/{}-to-{}/{}-to-{}_pv.maegz'.format(protein, target, start, target, start))
                    num_poses = len(list(structure.StructureReader(file)))
                    for i in range(1, num_poses):
                        if i > MAX_POSES:
                            break
                        with structure.StructureWriter(
                                '{}/{}/{}-to-{}/{}_lig{}.mae'.format(args.save_root, protein, target.lower(),
                                                                     start.lower(), target.lower(), str(i))) as all:
                            all.append(list(structure.StructureReader(file))[i])

if __name__=="__main__":
    main()