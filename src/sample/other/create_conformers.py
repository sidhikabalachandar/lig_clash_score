"""
The purpose of this code is to create conformers

It can be run on sherlock using
$ $SCHRODINGER/run python3 create_conformers.py check test /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits/search_test_incorrect_glide_index.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw --num_conformers 2 --index 0
"""

import argparse
import os
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
        for line in fp:
            if line[0] == '#':
                continue
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


def run_all(grouped_files, raw_root, run_path, docked_prot_file):
    for i in range(len(grouped_files)):
        cmd = 'sbatch -p owners -t 1:00:00 -o {} --wrap="$SCHRODINGER/run python3 create_conformers.py group {} {} {} ' \
              '--index {}"'
        os.system(cmd.format(os.path.join(run_path, 'conformer_{}.out'.format(i)), docked_prot_file, run_path, raw_root,
                             i))


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


def run_check(process, raw_root):
    unfinished = []
    for protein, target, start in process:
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, pair)
        if not os.path.exists(os.path.join(pair_path, '{}_lig0-out.maegz'.format(target))):
            unfinished.append((protein, target, start))

    print("Missing:", len(unfinished), "/", len(process))
    print(unfinished)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either align or search')
    parser.add_argument('mode', type=str, help='either train or test')
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('--index', type=int, default=-1, help='protein-ligand pair group index')
    parser.add_argument('--num_conformers', type=int, default=300, help='maximum number of conformers considered')
    parser.add_argument('--n', type=int, default=3, help='number of protein, target, start groups processed in '
                                                         'group task')
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
            run_all(grouped_files, args.raw_root, args.run_path, args.docked_prot_file)
        elif args.mode == 'test':
            process = get_prots(args.docked_prot_file)
            random.shuffle(process)
            for protein, target, start in process[:5]:
                print(protein, target, start)
                cmd = 'sbatch -p rondror -t 1:00:00 -o {} --wrap="$SCHRODINGER/run python3 create_conformers.py group ' \
                      'test {} {} {} --protein {} --target {} --start {}"'
                os.system(cmd.format(os.path.join(args.run_path, '{}_{}_{}.out'.format(protein, target, start)),
                                     args.docked_prot_file, args.run_path, args.raw_root, protein, target, start))

    elif args.task == 'group':
        if args.mode == 'train':
            # process = get_prots(args.docked_prot_file)
            process = [('P01901', '1fzk', '1fzo')]
            # grouped_files = group_files(args.n, process)
            for protein, target, start in process:
                pair = '{}-to-{}'.format(target, start)
                protein_path = os.path.join(args.raw_root, protein)
                pair_path = os.path.join(protein_path, pair)
                conformer_file = os.path.join(pair_path, "{}_lig0-out.maegz".format(target))
                if not os.path.exists(conformer_file):
                    run_group(protein, target, start, args.raw_root, args.num_conformers)
        elif args.mode == 'test':
            run_group(args.protein, args.target, args.start, args.raw_root, args.num_conformers)

    if args.task == 'check':
        if args.mode == 'train':
            process = get_prots(args.docked_prot_file)
            run_check(process, args.raw_root)
        elif args.mode == 'test':
            process = get_prots(args.docked_prot_file)
            random.shuffle(process)
            run_check(process[:5], args.raw_root)

if __name__ == "__main__":
    main()
