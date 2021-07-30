"""
The purpose of this code is to create conformers

It can be run on sherlock using
$ $SCHRODINGER/run python3 create_conformers.py test group /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits/search_test_incorrect_glide_index.txt /home/users/sidhikab/lig_clash_score/src/sample/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw --protein C8B467 --target 5ult --start 5uov --index 0 --n 1
"""

import argparse
import os
import random
import sys
sys.path.insert(1, 'util')
from util import *

_CONFGEN_CMD = "$SCHRODINGER/confgenx -WAIT -optimize -drop_problematic -num_conformers {num_conformers} " \
               "-max_num_conformers {num_conformers} {input_file}"

# MAIN TASK FUNCTIONS


def run_group(protein, target, start, args):
    # important dirs
    pair = '{}-to-{}'.format(target, start)
    protein_path = os.path.join(args.raw_root, protein)
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
    # copy ligand file
    print(f'cp {target_lig_file:} ./{basename:}')
    run_cmd(f'cp {target_lig_file:} ./{basename:}')
    command = _CONFGEN_CMD.format(num_conformers=args.num_conformers, input_file=f'./{basename:}')

    # run confgen
    print(command)
    run_cmd(command, f'Failed to run ConfGen on {target_lig_file:}')

    # remove copied file
    print(f'rm ./{basename:}')
    run_cmd(f'rm ./{basename:}')
    print('done')
    os.chdir(current_dir)

    if os.path.exists(os.path.join(pair_path, '{}_lig0.log'.format(target))):
        os.remove(os.path.join(pair_path, '{}_lig0.log'.format(target)))


def run_check(process, args):
    unfinished = []
    for protein, target, start in process:
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(args.raw_root, protein)
        pair_path = os.path.join(protein_path, pair)
        if not os.path.exists(os.path.join(pair_path, '{}_lig0-out.maegz'.format(target))):
            unfinished.append((protein, target, start))

    print("Missing:", len(unfinished), "/", len(process))
    print(unfinished)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, help='either train or test')
    parser.add_argument('task', type=str, help='either align or search')
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('--index', type=int, default=-1, help='protein-ligand pair group index')
    parser.add_argument('--num_conformers', type=int, default=300, help='maximum number of conformers considered')
    parser.add_argument('--n', type=int, default=1, help='number of protein, target, start groups processed in '
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
            for i in range(len(grouped_files)):
                cmd = 'sbatch -p owners -t 1:00:00 -o {} --wrap="$SCHRODINGER/run python3 create_conformers.py group ' \
                      'train {} {} {} --index {}"'
                os.system(
                    cmd.format(os.path.join(args.run_path, 'conformer_{}.out'.format(i)), args.docked_prot_file,
                               args.run_path, args.raw_root, i))
        elif args.mode == 'test':
            process = get_prots(args.docked_prot_file)
            random.shuffle(process)
            for protein, target, start in process[5:15]:
                print(protein, target, start)
                cmd = 'sbatch -p rondror -t 1:00:00 -o {} --wrap="$SCHRODINGER/run python3 create_conformers.py group ' \
                      'test {} {} {} --protein {} --target {} --start {}"'
                os.system(cmd.format(os.path.join(args.run_path, '{}_{}_{}.out'.format(protein, target, start)),
                                     args.docked_prot_file, args.run_path, args.raw_root, protein, target, start))
                return

    elif args.task == 'group':
        if args.mode == 'train':
            process = get_prots(args.docked_prot_file)
            grouped_files = group_files(args.n, process)
            for protein, target, start in grouped_files[args.index]:
                print(protein, target, start)
                pair = '{}-to-{}'.format(target, start)
                protein_path = os.path.join(args.raw_root, protein)
                pair_path = os.path.join(protein_path, pair)
                conformer_file = os.path.join(pair_path, "{}_lig0-out.maegz".format(target))
                if not os.path.exists(conformer_file):
                    run_group(protein, target, start, args)
        elif args.mode == 'test':
            run_group(args.protein, args.target, args.start, args)

    if args.task == 'check':
        if args.mode == 'train':
            process = get_prots(args.docked_prot_file)
            run_check(process, args)
        elif args.mode == 'test':
            process = get_prots(args.docked_prot_file)
            random.shuffle(process)
            run_check(process[5:15], args)

    if args.task == 'delete':
        process = get_prots(args.docked_prot_file)
        random.shuffle(process)
        for protein, target, start in process[5:15]:
            pair = '{}-to-{}'.format(target, start)
            protein_path = os.path.join(args.raw_root, protein)
            pair_path = os.path.join(protein_path, pair)
            conformer_file = os.path.join(pair_path, "{}_lig0-out.maegz".format(target))
            os.system('rm -rf {}'.format(conformer_file))

if __name__ == "__main__":
    main()
