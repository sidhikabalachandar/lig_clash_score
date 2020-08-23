"""
The purpose of this code is to create the .zip structure files
It can be run on sherlock using
$ ml load chemistry
$ ml load schrodinger
$ $SCHRODINGER/run python3 zipper.py /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
$ $SCHRODINGER/run python3 zipper.py check
"""

import os
from schrodinger.structure import StructureReader
from schrodinger.structutils.transform import get_centroid
from tqdm import tqdm
import argparse

def get_prots(fname):
    pairs = []
    with open(fname) as fp:
        for line in tqdm(fp, desc='files'):
            if line[0] == '#': continue
            protein, target, start = line.strip().split()
            pairs.append((protein, target, start))

    return pairs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('save', type=str, help='file listing proteins to process')
    parser.add_argument('raw_dir', type=str, help='file listing proteins to process')
    args = parser.parse_args()

    pairs = get_prots(args.prot_file)
    n = 3
    grouped_files = []

    for i in range(0, len(pairs), n):
        grouped_files += [pairs[i: i + n]]

    run_dir = os.path.join(args.save, 'run')
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)

    for i in range(len(grouped_files)):
        if i == 0:
            continue
        with open('{}/grid{}_in.sh'.format(run_dir, i), 'w') as f:
            for protein, target, start in grouped_files[i]:
                pair_dir = os.path.join(args.raw_dir, '{}/{}-to-{}'.format(protein, target, start))
                for file in os.listdir(pair_dir):
                    if '{}_lig'.format(target) in file and file[-3:] == 'mae':
                        script_dir = os.path.join(pair_dir, 'grid_scripts')
                        out_dir = os.path.join(pair_dir, 'grids')
                        if not os.path.exists(script_dir):
                            os.mkdir(script_dir)
                        if not os.path.exists(out_dir):
                            os.mkdir(out_dir)
                        with open('{}/{}.in'.format(script_dir, file[:-4]), 'w') as f_in:
                            s = next(StructureReader(os.path.join(pair_dir, file)))
                            c = get_centroid(s)
                            x, y, z = c[:3]

                            f_in.write('GRID_CENTER {},{},{}\n'.format(x, y, z))
                            f_in.write('GRIDFILE {}/{}.zip\n'.format(out_dir, file[:-4]))
                            f_in.write('INNERBOX 15,15,15\n')
                            f_in.write('OUTERBOX 30,30,30\n')
                            f_in.write('RECEP_FILE {}/{}_prot.mae\n'.format(pair_dir, start))
                            f.write('#!/bin/bash\n')
                            f.write('cd {}\n'.format(script_dir))
                            f.write('$SCHRODINGER/glide -WAIT {}/{}.in\n'.format(script_dir, file[:-4]))

        os.chdir(run_dir)
        os.system('sbatch -p owners -t 05:00:00 -o grid{}.out grid{}_in.sh'.format(i, i))

if __name__ == '__main__':
    main()


