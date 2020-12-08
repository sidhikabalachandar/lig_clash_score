"""
The purpose of this code is to create the cumulative frequency and bar graphs

It can be run on sherlock using
$ $SCHRODINGER/run python3 conformer_decoys.py all /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data_analysis/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /home/users/sidhikab/lig_clash_score/reports/figures/clash_analysis.png
$ $SCHRODINGER/run python3 conformer_decoys.py group /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data_analysis/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /home/users/sidhikab/lig_clash_score/reports/figures/clash_analysis.png --index 0
$ $SCHRODINGER/run python3 conformer_decoys.py check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data_analysis/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /home/users/sidhikab/lig_clash_score/reports/figures/clash_analysis.png
$ $SCHRODINGER/run python3 conformer_decoys.py graph /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data_analysis/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /home/users/sidhikab/lig_clash_score/reports/figures/clash_analysis.png
"""

import argparse
import os
import schrodinger.structure as structure
import schrodinger.structutils.interactions.steric_clash as steric_clash
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

def get_prots(docked_prot_file, raw_root):
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
            pair = '{}-to-{}'.format(target, start)
            protein_path = os.path.join(raw_root, protein)
            pair_path = os.path.join(protein_path, pair)
            pose_path = os.path.join(pair_path, 'conformer_poses')
            if len(os.listdir(pose_path)) == 100:
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

def run_all(run_path, grouped_files, docked_prot_file, raw_root):
    """
    submits sbatch script to create decoys for each protein, target, start group
    :param docked_prot_file: (string) file listing proteins to process
    :param run_path: (string) directory where script and output files will be written
    :param raw_root: (string) directory where raw data will be placed
    :param data_root: (string) pdbbind directory where raw data will be obtained
    :param grouped_files: (list) list of protein, target, start groups
    :return:
    """
    for i, group in enumerate(grouped_files):
        cmd = 'sbatch -p owners -t 1:00:00 -o {} --wrap="$SCHRODINGER/run python3 conformer_decoys.py group {} {} {} ' \
              '--index {}"'
        os.system(cmd.format(os.path.join(run_path, 'decoy{}.out'.format(i)), docked_prot_file, run_path, raw_root, i))

def run_group(grouped_files, raw_root, index, clash_dir):
    clashes = {}
    for protein, target, start in grouped_files[index]:
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, pair)
        pose_path = os.path.join(pair_path, 'conformer_poses')
        prot_file = os.path.join(pair_path, '{}_prot.mae'.format(start))
        prot = list(structure.StructureReader(prot_file))[0]

        clashes[(protein, target, start)] = []
        for i in range(100):
            decoy_file = os.path.join(pose_path, "{}_lig{}.mae".format(target, i))
            s = list(structure.StructureReader(decoy_file))[0]
            clashes[(protein, target, start)].append(steric_clash.clash_volume(prot, struc2=s))

    outfile = open(os.path.join(clash_dir, '{}.pkl'.format(index)), 'wb')
    pickle.dump(clashes, outfile)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either run, check, or update')
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('save_path', type=str, help='directory where graph will be saved')
    parser.add_argument('--index', type=int, default=-1, help='group index')
    parser.add_argument('--clash_dir', type=str, default=os.path.join(os.getcwd(), 'clash'), help='group index')
    parser.add_argument('--n', type=int, default=3, help='number of protein, target, start groups processed in '
                                                         'group task')
    args = parser.parse_args()

    if not os.path.exists(args.run_path):
        os.mkdir(args.run_path)

    if not os.path.exists(args.clash_dir):
        os.mkdir(args.clash_dir)

    if args.task == 'all':
        process = get_prots(args.docked_prot_file, args.raw_root)
        grouped_files = group_files(args.n, process)
        run_all(args.run_path, grouped_files, args.docked_prot_file, args.raw_root)

    if args.task == 'group':
        process = get_prots(args.docked_prot_file, args.raw_root)
        grouped_files = group_files(args.n, process)
        run_group(grouped_files, args.raw_root, args.index, args.clash_dir)

    if args.task == 'check':
        process = get_prots(args.docked_prot_file, args.raw_root)
        grouped_files = group_files(args.n, process)
        if len(os.listdir(args.clash_dir)) != len(grouped_files):
            print("Num expected: ", len(grouped_files))
            print("Num found: ", len(os.listdir(args.clash_dir)))
        else:
            print("Finished")

    if args.task == 'graph':
        clashes = {}
        # for protein, target, start in grouped_files[index]:
        protein = 'A0F7J4'
        target = '2rkf'
        start = '2rkg'
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(args. raw_root, protein)
        pair_path = os.path.join(protein_path, pair)
        pose_path = os.path.join(pair_path, 'conformer_poses')
        prot_file = os.path.join(pair_path, '{}_prot.mae'.format(start))
        prot = list(structure.StructureReader(prot_file))[0]

        clashes = []
        for i in range(100):
            decoy_file = os.path.join(pose_path, "{}_lig{}.mae".format(target, i))
            s = list(structure.StructureReader(decoy_file))[0]
            clashes.append(steric_clash.clash_volume(prot, struc2=s))
        # clashes = []
        # for file in os.listdir(args.clash_dir):
        #     infile = open(os.path.join(args.clash_dir, file), 'rb')
        #     clash = pickle.load(infile)
        #     infile.close()
        #     for lig in clash:
        #         clashes.extend(clash[lig])

        fig, ax = plt.subplots()
        sns.distplot(clashes, hist=False)
        plt.title('Clash Distributions for A0F7J4 2rkf-to-2rkg')
        plt.xlabel('clash volume')
        plt.ylabel('frequency')
        fig.savefig(args.save_path)

if __name__=="__main__":
    main()