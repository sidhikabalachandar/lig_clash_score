"""
The purpose of this code is to create the csv files with rmsd, mcss, and physcis score information

the code also combines all of the info for each protein, target, start group into one csv file

It can be run on sherlock using
ml load chemistry
ml load schrodinger
$ $SCHRODINGER/run python3 get_csv.py all /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d
$ $SCHRODINGER/run python3 get_csv.py group /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d --index 0
$ $SCHRODINGER/run python3 get_csv.py check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d
$ $SCHRODINGER/run python3 get_csv.py combine /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d
$ $SCHRODINGER/run python3 get_csv.py update /home/users/sidhikab/plep/index/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d --new_prot_file /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt
"""

import pandas as pd
import os
import argparse
from tqdm import tqdm

import sys
sys.path.append('/home/users/sidhikab/docking')
from docking.docking_class import Docking_Set
from docking.utilities import score_no_vdW

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

def to_df(data, out_dir, pair):
    '''
    write list of rmsds to csv file
    :param data: (list) list of rmsds
    :param out_dir: (string) path to output directory for this pair's csv file
    :param pair: (string) {target}-to-{start}
    :return: all_rmsds: (list) list of all rmsds
    '''
    df = pd.DataFrame(data, columns=['protein', 'start', 'target', 'rmsd', 'score_no_vdw', 'mcss'])
    df.to_csv(os.path.join(out_dir, pair + '.csv'))

def get_rmsd_results(rmsd_file_path):
    '''
    Get list of rmsds
    Format of csv file:
    "Index","Title","Mode","RMSD","Max dist.","Max dist atom index pair","ASL"
    "1","2W1I_pdb.smi:1","In-place","8.38264277875","11.5320975551","10-20","not atom.element H"
    :param rmsd_file_path: (string) rmsd file
    :return: all_rmsds: (list) list of all rmsds
    '''
    all_rmsds = []
    with open(rmsd_file_path) as rmsd_file:
        for line in list(rmsd_file)[1:]:
            rmsd = line.split(',')[3].strip('"')
            all_rmsds.append(float(rmsd))
    return all_rmsds

def run_all(docked_prot_file, run_path, raw_root, out_dir, grouped_files, n):
    for i, group in enumerate(grouped_files):
        cmd = 'sbatch -p owners -t 1:00:00 -o {} --wrap="$SCHRODINGER/run python3 get_csv.py group {} {} {} {} --n {}' \
              '--index {}"'
        os.system(cmd.format(os.path.join(run_path, 'labels{}.out'.format(i)), docked_prot_file,
                             run_path, raw_root, out_dir, i))

def run_group(grouped_files, raw_root, index):
    for protein, target, start in grouped_files[index]:
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, pair)
        pose_path = os.path.join(pair_path, 'ligand_poses')
        pair_data = []

        # get mcss
        with open('{}/{}_mcss.csv'.format(pair_path, pair)) as f:
            mcss = int(f.readline().strip().split(',')[4])

        # get rmsd
        rmsds = pd.read_csv('{}/{}_rmsd.csv'.format(pair_path, pair))

        # get physics score
        docking_config = [{'folder': pair_path,
                           'name': pair,
                           'grid_file': os.path.join(pair_path, '{}.zip'.format(pair)),
                           'prepped_ligand_file': os.path.join(pair_path, '{}_merge_pv.mae.gz'.format(pair)),
                           'glide_settings': {'num_poses': 1, 'docking_method': 'inplace'},
                           'ligand_file': os.path.join(pose_path, '{}_lig0.mae'.format(target))}]
        dock_set = Docking_Set()
        results = dock_set.get_docking_gscores(docking_config, mode='multi')
        results_by_ligand = results[pair]
        for file in results_by_ligand:
            score = score_no_vdW(results_by_ligand[file][0])
            rmsd = rmsds[rmsds['Title'] == file]['RMSD'].iloc[0]
            pair_data.append([protein, start, file[:-4], rmsd, score, mcss])

        to_df(pair_data, pair_path, pair)
        # os.remove(os.path.join(pair_path, '{}_mcss.csv'.format(pair)))
        # os.remove(os.path.join(pair_path, '{}_mege_pv.mae.gz'.format(pair)))
        if os.path.exists(os.path.join(pair_path, '{}.in'.format(pair))):
            os.remove(os.path.join(pair_path, '{}.in'.format(pair)))
        if os.path.exists(os.path.join(pair_path, '{}.log'.format(pair))):
            os.remove(os.path.join(pair_path, '{}.log'.format(pair)))
        if os.path.exists(os.path.join(pair_path, '{}_pv.maegz'.format(pair))):
            os.remove(os.path.join(pair_path, '{}_pv.maegz'.format(pair)))
        # os.remove(os.path.join(pair_path, '{}_rmsd.csv'.format(pair)))
        # os.remove(os.path.join(pair_path, '{}.scor'.format(pair)))
        # os.remove(os.path.join(pair_path, '{}.zip'.format(pair)))
        if os.path.exists(os.path.join(pair_path, '{}_state.json'.format(pair))):
            os.remove(os.path.join(pair_path, '{}_state.json'.format(pair)))

def run_check(docked_prot_file, raw_root):
    process = []
    num_pairs = 0
    with open(docked_prot_file) as fp:
        for line in tqdm(fp, desc='going through protein, target, start groups'):
            if line[0] == '#': continue
            protein, target, start = line.strip().split()
            num_pairs += 1
            protein_path = os.path.join(raw_root, protein)
            pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
            if not os.path.exists(os.path.join(pair_path, '{}-to-{}.csv'.format(target, start))):
                process.append((protein, target, start))

    print('Missing', len(process), '/', num_pairs)
    print(process)

def run_combine(docked_prot_file, raw_root, out_dir):
    dfs = []
    with open(docked_prot_file) as fp:
        for line in tqdm(fp, desc='going through protein, target, start groups'):
            if line[0] == '#': continue
            protein, target, start = line.strip().split()
            protein_path = os.path.join(raw_root, protein)
            pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
            dfs.append(pd.read_csv(os.path.join(pair_path, '{}-to-{}.csv'.format(target, start))))

    combined_csv_data = pd.concat(dfs)
    combined_csv_data.to_csv(os.path.join(out_dir, 'combined.csv'))

def update(docked_prot_file, raw_root, new_prot_file):
    """
    update index by removing protein, target, start that could not create grids
    :param docked_prot_file: (string) file listing proteins to process
    :param raw_root: (string) directory where raw data will be placed
    :param new_prot_file: (string) name of new prot file
    :return:
    """
    text = []
    with open(docked_prot_file) as fp:
        for line in tqdm(fp, desc='files'):
            if line[0] == '#': continue
            protein, target, start = line.strip().split()
            protein_path = os.path.join(raw_root, protein)
            pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
            if os.path.exists(os.path.join(pair_path, '{}-to-{}.csv'.format(target, start))):
                text.append(line)

    file = open(new_prot_file, "w")
    file.writelines(text)
    file.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either all, group, check, combine, or update')
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('out_dir', type=str, help='directory where the combined rmsd csv file will be placed')
    parser.add_argument('--index', type=int, default=-1, help='for group task, group number')
    parser.add_argument('--n', type=int, default=3, help='number of protein, target, start groups processed in '
                                                         'group task')
    parser.add_argument('--new_prot_file', type=str, default=os.path.join(os.getcwd(), 'index.txt'),
                        help='for update task, name of new prot file')
    args = parser.parse_args()

    if not os.path.exists(args.run_path):
        os.mkdir(args.run_path)

    if args.task == 'all':
        process = get_prots(args.docked_prot_file)
        grouped_files = group_files(args.n, process)
        run_all(args.docked_prot_file, args.run_path, args.raw_root, args.out_dir, grouped_files, args.n)

    if args.task == 'group':
        process = get_prots(args.docked_prot_file)
        grouped_files = group_files(args.n, process)
        run_group(grouped_files, args.raw_root, args.index)

    if args.task == 'check':
        run_check(args.docked_prot_file, args.raw_root)

    if args.task == 'combine':
        run_combine(args.docked_prot_file, args.raw_root, args.out_dir)

    if args.task == 'update':
        update(args.docked_prot_file, args.raw_root, args.new_prot_file)

if __name__ == "__main__":
    main()



