"""
The purpose of this code is to create conformers

It can be run on sherlock using
$ $SCHRODINGER/run python3 shape_align_conformers.py test delete /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits/search_test_incorrect_glide_index.txt /home/users/sidhikab/lig_clash_score/src/sample/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw --index 0 --n 1
"""

import argparse
import os
import schrodinger.structure as structure
import schrodinger.structutils.build as build
import random
import sys
sys.path.insert(1, 'util')
from util import *

_ALIGN_CMD = "$SCHRODINGER/shape_screen -shape {shape} -screen {screen} -WAIT -JOB {job_name}"

# MAIN TASK FUNCTIONS


def run_group(protein, target, start, args):
    print(protein, target, start)
    # important dirs
    pair = '{}-to-{}'.format(target, start)
    protein_path = os.path.join(args.raw_root, protein)
    pair_path = os.path.join(protein_path, pair)
    output_path = os.path.join(pair_path, 'conformers')

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # get conformers
    conformer_file = os.path.join(pair_path, "{}_lig0-out.maegz".format(target))
    conformers = list(structure.StructureReader(conformer_file))
    num = min(args.num_conformers, len(conformers))

    # align each conformer to starting ligand
    for i in range(num):
        aligned_conformer_file = os.path.join(output_path, '{}_align.maegz'.format(i))
        if not os.path.exists(aligned_conformer_file):
            os.chdir(output_path)
            screen_file = os.path.join(output_path, "screen_{}.mae".format(i))
            with structure.StructureWriter(screen_file) as screen:
                screen.append(conformers[i])

            shape_file = os.path.join(pair_path, '{}_lig.mae'.format(start))
            # run alignment
            run_cmd(_ALIGN_CMD.format(shape=shape_file, screen=screen_file, job_name=i))

        # align with hydrogen
        aligned_conformer = list(structure.StructureReader(aligned_conformer_file))[0]
        hydrogen_file = os.path.join(output_path, "{}_align_with_hydrogen.mae".format(i))
        if not os.path.exists(hydrogen_file):
            with structure.StructureWriter(hydrogen_file) as h:
                h.append(aligned_conformer)

        # align without hydrogen
        no_hydrogen_file = os.path.join(output_path, "{}_align_without_hydrogen.mae".format(i))
        if not os.path.exists(no_hydrogen_file):
            build.delete_hydrogens(aligned_conformer)

            with structure.StructureWriter(no_hydrogen_file) as no_h:
                no_h.append(aligned_conformer)

    # combine all ligands with hydrogen
    combined_file = os.path.join(pair_path, "aligned_to_start_with_hydrogen_conformers.mae")
    with structure.StructureWriter(combined_file) as combined:
        for i in range(num):
            aligned_file = os.path.join(output_path, '{}_align_with_hydrogen.mae'.format(i))
            s = list(structure.StructureReader(aligned_file))[0]
            combined.append(s)

    # combine all ligands without hydrogen
    combined_file = os.path.join(pair_path, "aligned_to_start_without_hydrogen_conformers.mae")
    with structure.StructureWriter(combined_file) as combined:
        for i in range(num):
            aligned_file = os.path.join(output_path, '{}_align_without_hydrogen.mae'.format(i))
            s = list(structure.StructureReader(aligned_file))[0]
            combined.append(s)


def run_check(conformer_prots, args):
    unfinished = []
    for protein, target, start in conformer_prots:
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(args.raw_root, protein)
        pair_path = os.path.join(protein_path, pair)

        conformer_file = os.path.join(pair_path, "{}_lig0-out.maegz".format(target))
        conformers = list(structure.StructureReader(conformer_file))
        output_path = os.path.join(pair_path, 'conformers')

        num = min(args.num_conformers, len(conformers))
        for i in range(num):
            if not os.path.exists(os.path.join(output_path, '{}_align_without_hydrogen.mae'.format(i))) or \
                    not os.path.exists(os.path.join(output_path, '{}_align_with_hydrogen.mae'.format(i))):
                unfinished.append((protein, target, start, i))
                break

    print("Missing: {} / {}".format(len(unfinished), len(conformer_prots)))
    print(unfinished)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, help='either train or test')
    parser.add_argument('task', type=str, help='either align or search')
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('--n', type=int, default=10, help='number of alignments processed in each job')
    parser.add_argument('--index', type=int, default=-1, help='grid point group index')
    parser.add_argument('--num_conformers', type=int, default=300, help='maximum number of conformers considered')
    parser.add_argument('--protein', type=str, default='', help='name of protein')
    parser.add_argument('--target', type=str, default='', help='name of target ligand')
    parser.add_argument('--start', type=str, default='', help='name of start ligand')
    args = parser.parse_args()

    random.seed(0)

    if not os.path.exists(args.run_path):
        os.mkdir(args.run_path)

    if args.task == 'all':
        if args.mode == 'train':
            process = get_prots(args.docked_prot_file)
            grouped_files = group_files(args.n, process)
            for i in range(len(grouped_files)):
                cmd = 'sbatch -p owners -t 1:00:00 -o {} --wrap="$SCHRODINGER/run python3 shape_align_conformers.py ' \
                      'train group {} {} {} --n {} --num_conformers {} --index {}"'
                os.system(
                    cmd.format(os.path.join(args.run_path, 'align_{}.out'.format(i)), args.docked_prot_file, args.run_path,
                               args.raw_root, args.n, args.num_conformers, i))
        elif args.mode == 'test':
            process = get_prots(args.docked_prot_file)
            random.shuffle(process)
            for protein, target, start in process[:5]:
                pair = '{}-to-{}'.format(target, start)
                protein_path = os.path.join(args.raw_root, protein)
                pair_path = os.path.join(protein_path, pair)
                conformer_file = os.path.join(pair_path, "{}_lig0-out.maegz".format(target))
                conformers = list(structure.StructureReader(conformer_file))
                num = min(args.num_conformers, len(conformers))
                ls = [i for i in range(num)]
                grouped_files = group_files(args.n, ls)
                for i in range(len(grouped_files)):
                    cmd = 'sbatch -p owners -t 1:00:00 -o {} --wrap="$SCHRODINGER/run python3 ' \
                          'shape_align_conformers.py test group {} {} {} --protein {} --target {} --start {} --index ' \
                          '{}"'
                    os.system(
                        cmd.format(os.path.join(args.run_path, '{}_{}_{}_{}.out'.format(protein, target, start, i)),
                                   args.docked_prot_file, args.run_path, args.raw_root, protein, target, start, i))

    elif args.task == 'group':
        if args.mode == 'train':
            process = get_prots(args.docked_prot_file)
            grouped_files = group_files(args.n, process)
            for protein, target, start in grouped_files[index]:
                run_group(protein, target, start, args)
        elif args.mode == 'test':
            run_group(args.protein, args.target, args.start, args)

    elif args.task == 'check':
        if args.mode == 'train':
            process = get_prots(args.docked_prot_file)
            run_check(process, args)
        elif args.mode == 'test':
            process = get_prots(args.docked_prot_file)
            random.shuffle(process)
            run_check(process[:5], args)

    elif args.task == 'delete':
        process = get_prots(args.docked_prot_file)
        random.shuffle(process)
        for protein, target, start in process[5:15]:
            pair = '{}-to-{}'.format(target, start)
            protein_path = os.path.join(args.raw_root, protein)
            pair_path = os.path.join(protein_path, pair)
            output_path = os.path.join(pair_path, 'conformers')

            os.system('rm -rf {}'.format(output_path))


if __name__ == "__main__":
    main()
