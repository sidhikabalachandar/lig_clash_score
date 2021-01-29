"""
The purpose of this code is to create conformers

It can be run on sherlock using
$ $SCHRODINGER/run python3 shape_align_conformers.py combine /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw --protein P07900 --target 2qg2 --start 1yet --index 1
"""

import argparse
import os
import schrodinger.structure as structure
import schrodinger.structutils.build as build
import subprocess
import random
from tqdm import tqdm

_ALIGN_CMD = "$SCHRODINGER/shape_screen -shape {shape} -screen {screen} -WAIT -JOB {job_name}"

# HELPER FUNCTIONS


def get_prots(docked_prot_file):
    """
    gets list of all protein, target ligands, and starting ligands in the index file
    :param docked_prot_file: (string) file listing proteins to process
    :return: process (list) list of all protein, target ligands, and starting ligands to process
    """
    process = []
    with open(docked_prot_file) as fp:
        for line in tqdm(fp, desc='index file'):
            if line[0] == '#':
                continue
            protein, target, start = line.strip().split()
            process.append((protein, target, start))

    return process


def group_conformers(n, protein, target, start, raw_root):
    pair = '{}-to-{}'.format(target, start)
    protein_path = os.path.join(raw_root, protein)
    pair_path = os.path.join(protein_path, pair)
    conformer_file = os.path.join(pair_path, "{}_lig0-out.maegz".format(target))
    conformers = list(structure.StructureReader(conformer_file))
    grouped_files = []

    for i in range(0, len(conformers), n):
        grouped_files.append(conformers[i: i + n])

    return grouped_files


def get_conformer_prots(process, raw_root, num_pairs):
    conformer_prots = []
    for protein, target, start in process:
        if len(conformer_prots) == num_pairs:
            break
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, pair)

        conformer_file = os.path.join(pair_path, "{}_lig0-out.maegz".format(target))
        conformers = list(structure.StructureReader(conformer_file))
        if len(conformers) == 1:
            continue
        else:
            conformer_prots.append((protein, target, start))

    return conformer_prots


def run_cmd(cmd, error_msg=None, raise_except=False):
    try:
        return subprocess.check_output(
            cmd,
            universal_newlines=True,
            shell=True)
    except Exception as e:
        if error_msg is not None:
            print(error_msg)
        if raise_except:
            raise e

# MAIN TASK FUNCTIONS


def run_all(conformer_prots, raw_root, run_path, docked_prot_file, n):
    for protein, target, start in conformer_prots:
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, pair)
        output_path = os.path.join(pair_path, 'conformers')

        # if not os.path.exists(os.path.join(output_path, '0_align_without_hydrogen.mae')) or \
        #         not os.path.exists(os.path.join(output_path, '0_align_with_hydrogen.mae')):
        grouped_files = group_conformers(n, protein, target, start, raw_root)
        for i, group in enumerate(grouped_files):
            cmd = 'sbatch -p owners -t 1:00:00 -o {} --wrap="$SCHRODINGER/run python3 shape_align_conformers.py ' \
                  'group {} {} {} --n {} --index {} --protein {} --target {} --start {}"'
            os.system(cmd.format(os.path.join(run_path, 'decoy{}.out'.format(i)), docked_prot_file, run_path,
                                 raw_root, n, i, protein, target, start))


def run_group(grouped_files, index, n, protein, target, start, raw_root):
    for i, conformer in enumerate(grouped_files[index]):
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, pair)
        output_path = os.path.join(pair_path, 'conformers')

        if not os.path.exists(output_path):
            os.mkdir(output_path)

        num = str(n * index + i)
        aligned_conformer_file = os.path.join(output_path, '{}_align.maegz'.format(num))
        if not os.path.exists(aligned_conformer_file):
            os.chdir(output_path)
            screen_file = os.path.join(output_path, "screen_{}.mae".format(num))
            with structure.StructureWriter(screen_file) as screen:
                screen.append(conformer)

            shape_file = os.path.join(pair_path, '{}_lig.mae'.format(start))
            run_cmd(_ALIGN_CMD.format(shape=shape_file, screen=screen_file, job_name=num))

        aligned_conformer = list(structure.StructureReader(aligned_conformer_file))[0]
        hydrogen_file = os.path.join(output_path, "{}_align_with_hydrogen.mae".format(num))
        if not os.path.exists(hydrogen_file):
            with structure.StructureWriter(hydrogen_file) as h:
                h.append(aligned_conformer)

        no_hydrogen_file = os.path.join(output_path, "{}_align_without_hydrogen.mae".format(num))
        if not os.path.exists(no_hydrogen_file):
            build.delete_hydrogens(aligned_conformer)

            with structure.StructureWriter(no_hydrogen_file) as no_h:
                no_h.append(aligned_conformer)


def run_check(conformer_prots, raw_root):
    unfinished = []
    for protein, target, start in conformer_prots:
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, pair)

        conformer_file = os.path.join(pair_path, "{}_lig0-out.maegz".format(target))
        conformers = list(structure.StructureReader(conformer_file))
        output_path = os.path.join(pair_path, 'conformers')

        for i in range(len(conformers)):
            if not os.path.exists(os.path.join(output_path, '{}_align_without_hydrogen.mae'.format(i))) or \
                    not os.path.exists(os.path.join(output_path, '{}_align_with_hydrogen.mae'.format(i))):
                unfinished.append((protein, target, start, i))

    print("Missing:", len(unfinished))
    print(unfinished)


def run_combine(conformer_prots, raw_root):
    for protein, target, start in conformer_prots:
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, pair)
        conformer_file = os.path.join(pair_path, "{}_lig0-out.maegz".format(target))
        conformers = list(structure.StructureReader(conformer_file))
        output_path = os.path.join(pair_path, 'conformers')

        combined_file = os.path.join(pair_path, "aligned_to_start_without_hydrogen_conformers.mae")
        if not os.path.exists(combined_file):
            with structure.StructureWriter(combined_file) as combined:
                for i in range(len(conformers)):
                    aligned_file = os.path.join(output_path, '{}_align_without_hydrogen.mae'.format(i))
                    s = list(structure.StructureReader(aligned_file))[0]
                    combined.append(s)

        combined_file = os.path.join(pair_path, "aligned_to_start_with_hydrogen_conformers.mae")
        if not os.path.exists(combined_file):
            with structure.StructureWriter(combined_file) as combined:
                for i in range(len(conformers)):
                    aligned_file = os.path.join(output_path, '{}_align_with_hydrogen.mae'.format(i))
                    s = list(structure.StructureReader(aligned_file))[0]
                    combined.append(s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either align or search')
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('--protein', type=str, default='', help='protein name')
    parser.add_argument('--target', type=str, default='', help='target ligand name')
    parser.add_argument('--start', type=str, default='', help='start ligand name')
    parser.add_argument('--num_conformers', type=int, default=300, help='maximum number of conformers considered')
    parser.add_argument('--num_pairs', type=int, default=10, help='number of protein-ligand pairs considered')
    parser.add_argument('--n', type=int, default=10, help='number of alignments processed in each job')
    parser.add_argument('--index', type=int, default=-1, help='grid point group index')
    args = parser.parse_args()

    random.seed(0)

    if not os.path.exists(args.run_path):
        os.mkdir(args.run_path)

    if args.task == 'all':
        # process = get_prots(args.docked_prot_file)
        # random.shuffle(process)
        # conformer_prots = get_conformer_prots(process, args.raw_root, args.num_pairs)
        conformer_prots = [('A5HZZ9', '3c8a', '3qw5'), ('O38732', '2i0a', '2q5k'), ('P07900', '2qg2', '1yet'),
                           ('P51449', '5vb6', '5ufr')]
        run_all(conformer_prots, args.raw_root, args.run_path, args.docked_prot_file, args.n)

    elif args.task == 'group':
        grouped_files = group_conformers(args.n, args.protein, args.target, args.start, args.raw_root)
        run_group(grouped_files, args.index, args.n, args.protein, args.target, args.start, args.raw_root)

    elif args.task == 'check':
        # process = get_prots(args.docked_prot_file)
        # random.shuffle(process)
        # conformer_prots = get_conformer_prots(process, args.raw_root, args.num_pairs)
        conformer_prots = [('A5HZZ9', '3c8a', '3qw5'), ('O38732', '2i0a', '2q5k'), ('P07900', '2qg2', '1yet'),
                           ('P51449', '5vb6', '5ufr')]
        run_check(conformer_prots, args.raw_root)

    elif args.task == 'combine':
        # process = get_prots(args.docked_prot_file)
        # random.shuffle(process)
        # conformer_prots = get_conformer_prots(process, args.raw_root, args.num_pairs)
        conformer_prots = [('A5HZZ9', '3c8a', '3qw5'), ('O38732', '2i0a', '2q5k'), ('P07900', '2qg2', '1yet'),
                           ('P51449', '5vb6', '5ufr')]
        run_combine(conformer_prots, args.raw_root)


if __name__ == "__main__":
    main()
