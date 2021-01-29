"""
The purpose of this code is to create conformers

It can be run on sherlock using
$ $SCHRODINGER/run python3 create_conformers.py check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw --protein P18031 --target 1g7g --start 1c83
"""

import argparse
import os
import schrodinger.structure as structure
import subprocess
import random
from tqdm import tqdm

_CONFGEN_CMD = "$SCHRODINGER/confgenx -WAIT -optimize -drop_problematic -num_conformers {num_conformers} " \
               "-max_num_conformers {num_conformers} {input_file}"

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


def run_all(process, raw_root, run_path, docked_prot_file, num_pairs):
    counter = 0
    for protein, target, start in process:
        if counter == num_pairs:
            break
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, pair)
        conformer_file = os.path.join(pair_path, "{}_lig0-out.maegz".format(target))
        if not os.path.exists(conformer_file):
            cmd = 'sbatch -p owners -t 1:00:00 -o {} --wrap="$SCHRODINGER/run python3 create_conformers.py group {} ' \
                  '{} {} --protein {} --target {} --start {}"'
            os.system(cmd.format(os.path.join(run_path, 'conformer_{}_{}-to-{}.out'.format(protein, target, start)),
                                 docked_prot_file, run_path, raw_root, protein, target, start))
            counter += 1
        else:
            conformers = list(structure.StructureReader(conformer_file))
            if len(conformers) > 1:
                counter += 1


def run_group(protein, target, start, raw_root, num_conformers):
    pair = '{}-to-{}'.format(target, start)
    protein_path = os.path.join(raw_root, protein)
    pair_path = os.path.join(protein_path, pair)
    target_lig_file = os.path.join(pair_path, 'ligand_poses', '{}_lig0.mae'.format(target))
    current_dir = os.getcwd()
    os.chdir(pair_path)
    basename = os.path.basename(target_lig_file)

    ### Note: For some reason, confgen isn't able to find the .mae file,
    # unless it is in working directory. So, we need to copy it over.
    ### Note: There may be duplicated ligand names (for different targets).
    # Since it only happens for CHEMBL ligand, just ignore it for now.
    # Otherwise, might want consider to generate the conformers to separate
    # folders for each (target, ligand) pair.
    # Run ConfGen
    run_cmd(f'cp {target_lig_file:} ./{basename:}')
    command = _CONFGEN_CMD.format(num_conformers=num_conformers,
                                  input_file=f'./{basename:}')
    run_cmd(command, f'Failed to run ConfGen on {target_lig_file:}')
    run_cmd(f'rm ./{basename:}')
    os.chdir(current_dir)

    if os.path.exists(os.path.join(pair_path, '{}_lig0.log'.format(target))):
        os.remove(os.path.join(pair_path, '{}_lig0.log'.format(target)))


def run_check(process, raw_root, num_pairs):
    unfinished = []
    counter = 0
    for protein, target, start in process:
        print(protein, target, start)
        if counter == num_pairs:
            break
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, pair)
        if not os.path.exists(os.path.join(pair_path, '{}_lig0-out.maegz'.format(target))):
            unfinished.append((protein, target, start))
        else:
            conformer_file = os.path.join(pair_path, "{}_lig0-out.maegz".format(target))
            conformers = list(structure.StructureReader(conformer_file))
            print(protein, target, start, len(conformers), counter)
            if len(conformers) > 1:
                counter += 1

    print("Missing:", len(unfinished), "/", num_pairs)
    print(unfinished)


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
    args = parser.parse_args()

    random.seed(0)

    if not os.path.exists(args.run_path):
        os.mkdir(args.run_path)

    if args.task == 'all':
        # process = get_prots(args.docked_prot_file)
        # random.shuffle(process)
        process = [('A5HZZ9', '3c8a', '3qw5'), ('O38732', '2i0a', '2q5k'), ('P07900', '2qg2', '1yet'),
                   ('P51449', '5vb6', '5ufr')]
        run_all(process, args.raw_root, args.run_path, args.docked_prot_file, args.num_pairs)

    elif args.task == 'group':
        run_group(args.protein, args.target, args.start, args.raw_root, args.num_conformers)

    if args.task == 'check':
        # process = get_prots(args.docked_prot_file)
        # random.shuffle(process)
        process = [('A5HZZ9', '3c8a', '3qw5'), ('O38732', '2i0a', '2q5k'), ('P07900', '2qg2', '1yet'),
                   ('P51449', '5vb6', '5ufr')]
        run_check(process, args.raw_root, args.num_pairs)


if __name__ == "__main__":
    main()
