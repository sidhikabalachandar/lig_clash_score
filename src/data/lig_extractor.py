"""
The purpose of this code is to set the train, val, and test data sets
It can be run on sherlock using
ml load chemistry
ml load schrodinger
$ $SCHRODINGER/run python3 lig_extractor.py all
$ $SCHRODINGER/run python3 lig_extractor.py group <index>
"""

import argparse
import os
import sys
import schrodinger.structure as structure

prot_file = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt'
save_root = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw'
run_path = '/home/users/sidhikab/flexibility_project/atom3d/src/data/run'
MAX_POSES = 100

def main():
    task = sys.argv[1]
    process = []
    counter = 0
    with open(prot_file) as fp:
        for line in fp:
            if line[0] == '#': continue
            counter += 1
            protein, target, start = line.strip().split()
            pv_file = os.path.join(save_root,
                                   '{}/{}-to-{}/{}-to-{}_pv.maegz'.format(protein, target, start, target, start))
            if os.path.exists(pv_file):
                max_num_poses = min(len(list(structure.StructureReader(pv_file))) - 1, MAX_POSES)
                lig_file = '{}/{}/{}-to-{}/{}_lig{}.mae'.format(save_root, protein, target, start, target, max_num_poses)
                if not os.path.exists(lig_file):
                    process.append(pv_file)

    print('Missing:', len(process), '/', counter)
    print(process)

    grouped_files = []
    n = 20

    for i in range(0, len(process), n):
        grouped_files += [process[i: i + n]]

    if not os.path.exists(run_path):
        os.mkdir(run_path)

    if task == 'all':
        for i, group in enumerate(grouped_files):
            cmd = 'sbatch -p owners -t 1:00:00 -o {} --wrap="$SCHRODINGER/run python3 lig_extractor.py group {}"'
            os.system(cmd.format(os.path.join(run_path, 'lig{}.out'.format(i)), i))
            # print(cmd.format(os.path.join(run_path, 'lig{}.out'.format(i)), i))

    if task == 'group':
        i = int(sys.argv[2])

        for file in grouped_files[i]:
            protein = file.split('/')[-3]
            [target, start] = file.split('/')[-2].split('-to-')
            num_poses = len(list(structure.StructureReader(file)))
            for i in range(1, num_poses):
                if i == 101:
                    break
                with structure.StructureWriter('{}/{}/{}-to-{}/{}_lig{}.mae'.format(save_root, protein, target, start, target, str(i))) as all:
                    all.append(list(structure.StructureReader(file))[i])

if __name__=="__main__":
    main()