"""
The purpose of this code is to create the csv files with rmsd, mcss, and physcis score information

the code also combines all of the info for each protein, target, start group into one csv file

It can be run on sherlock using
ml load chemistry
ml load schrodinger
$ $SCHRODINGER/run python3 get_csv.py all /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d

$ $SCHRODINGER/run python3 get_csv.py combine /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits/combined_index_balance_clash_large.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d --decoy_type conformer_poses

$ $SCHRODINGER/run python3 get_csv.py group /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d --index 0
$ $SCHRODINGER/run python3 get_csv.py check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d
$ $SCHRODINGER/run python3 get_csv.py combine /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d
$ $SCHRODINGER/run python3 get_csv.py update /home/users/sidhikab/plep/index/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d --new_prot_file /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt
$ $SCHRODINGER/run python3 get_csv.py remove /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d
$ $SCHRODINGER/run python3 get_csv.py check_ordering /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d
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

def get_glide_score(pdb, label_df):
    """
    searches for pdb's rmsd in combined rmsd df
    :param pdb: (string) {target}_lig{id}
    :param label_df: (df) combined rmsd df
    :return: (float) rmsd value
    """
    return label_df[label_df['target'] == pdb]['glide_score'].iloc[0]

def to_df(data, out_dir, pair, decoy_type):
    '''
    write list of rmsds to csv file
    :param data: (list) list of rmsds
    :param out_dir: (string) path to output directory for this pair's csv file
    :param pair: (string) {target}-to-{start}
    :return: all_rmsds: (list) list of all rmsds
    '''
    df = pd.DataFrame(data, columns=['protein', 'start', 'target', 'rmsd', 'modified_rmsd', 'mcss',
                                     'target_start_glide_score', 'target_start_score_no_vdw'])
    df.to_csv(os.path.join(out_dir, '{}_{}.csv'.format(pair, decoy_type)))

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

def run_all(docked_prot_file, run_path, raw_root, out_dir, grouped_files, n, decoy_type):
    for i, group in enumerate(grouped_files):
        cmd = 'sbatch -p owners -t 1:00:00 -o {} --wrap="$SCHRODINGER/run python3 get_csv.py group {} {} {} {} --n {} ' \
              '--index {} --decoy_type {}"'
        os.system(cmd.format(os.path.join(run_path, 'labels{}.out'.format(i)), docked_prot_file,
                             run_path, raw_root, out_dir, n, i, decoy_type))

def run_group(grouped_files, raw_root, index, rmsd_cutoff, decoy_type):
    for protein, target, start in grouped_files[index]:
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, pair)
        print(pair_path)
        pose_path = os.path.join(pair_path, decoy_type)
        pair_data = []

        # get mcss
        with open('{}/{}_mcss.csv'.format(pair_path, pair)) as f:
            mcss = int(f.readline().strip().split(',')[4])

        # get rmsd
        rmsds = pd.read_csv('{}/{}_{}_rmsd.csv'.format(pair_path, pair, decoy_type))

        # get physics score
        docking_config = [{'folder': pair_path,
                                   'name': '{}_{}'.format(pair, decoy_type),
                                   'grid_file': os.path.join(pair_path, '{}.zip'.format(pair)),
                                   'prepped_ligand_file': os.path.join(pair_path, '{}_{}_merge_pv.mae'.format(pair, decoy_type)),
                                   'glide_settings': {'num_poses': 1, 'docking_method': 'inplace'},
                                   'ligand_file': os.path.join(pose_path, '{}_lig0.mae'.format(target))}]
        dock_set = Docking_Set()
        results = dock_set.get_docking_gscores(docking_config, mode='multi')
        for file in results['{}_{}'.format(pair, decoy_type)]:
            target_start_results = results['{}_{}'.format(pair, decoy_type)]
            target_start_glide_score = target_start_results[file][0]['Score']
            target_start_score_no_vdw = score_no_vdW(target_start_results[file][0])
            rmsd = rmsds[rmsds['Title'] == file]['RMSD'].iloc[0]
            if rmsd > rmsd_cutoff:
                modified_rmsd = rmsd ** 3
            else:
                modified_rmsd = rmsd
            pair_data.append([protein, start, file[:-4], rmsd, modified_rmsd, mcss, target_start_glide_score,
                              target_start_score_no_vdw])

        to_df(pair_data, pair_path, pair, decoy_type)
        # os.remove(os.path.join(pair_path, '{}_mcss.csv'.format(pair)))
        # os.remove(os.path.join(pair_path, '{}_mege_pv.mae.gz'.format(pair)))
        if os.path.exists(os.path.join(pair_path, '{}_{}.in'.format(pair, decoy_type))):
            os.remove(os.path.join(pair_path, '{}_{}.in'.format(pair, decoy_type)))
        if os.path.exists(os.path.join(pair_path, '{}_{}.log'.format(pair, decoy_type))):
            os.remove(os.path.join(pair_path, '{}_{}.log'.format(pair, decoy_type)))
        if os.path.exists(os.path.join(pair_path, '{}_{}_pv.maegz'.format(pair, decoy_type))):
            os.remove(os.path.join(pair_path, '{}_{}_pv.maegz'.format(pair, decoy_type)))
        # os.remove(os.path.join(pair_path, '{}_rmsd.csv'.format(pair)))
        # os.remove(os.path.join(pair_path, '{}.scor'.format(pair)))
        # os.remove(os.path.join(pair_path, '{}.zip'.format(pair)))
        if os.path.exists(os.path.join(pair_path, '{}_{}_state.json'.format(pair, decoy_type))):
            os.remove(os.path.join(pair_path, '{}_{}_state.json'.format(pair, decoy_type)))

def run_check(docked_prot_file, raw_root, decoy_type):
    process = []
    num_pairs = 0
    with open(docked_prot_file) as fp:
        for line in tqdm(fp, desc='going through protein, target, start groups'):
            if line[0] == '#': continue
            protein, target, start = line.strip().split()
            num_pairs += 1
            protein_path = os.path.join(raw_root, protein)
            pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
            if not os.path.exists(os.path.join(pair_path, '{}-to-{}_{}.csv'.format(target, start, decoy_type))):
                process.append((protein, target, start))

    print('Missing', len(process), '/', num_pairs)
    print(process)

def run_combine(docked_prot_file, raw_root, out_dir, decoy_type):
    dfs = []
    with open(docked_prot_file) as fp:
        for line in tqdm(fp, desc='going through protein, target, start groups'):
            if line[0] == '#': continue
            protein, target, start = line.strip().split()
            protein_path = os.path.join(raw_root, protein)
            pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
            dfs.append(pd.read_csv(os.path.join(pair_path, '{}-to-{}_{}.csv'.format(target, start, decoy_type))))

    combined_csv_data = pd.concat(dfs)
    combined_csv_data.to_csv(os.path.join(out_dir, 'combined_{}.csv'.format(decoy_type)))

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
    parser.add_argument('--n', type=int, default=10, help='number of protein, target, start groups processed in '
                                                         'group task')
    parser.add_argument('--new_prot_file', type=str, default=os.path.join(os.getcwd(), 'index.txt'),
                        help='for update task, name of new prot file')
    parser.add_argument('--max_poses', type=int, default=100, help='maximum number of glide poses considered')
    parser.add_argument('--rmsd_cutoff', type=int, default=2, help='maximum number of glide poses considered')
    parser.add_argument('--decoy_type', type=str, default='ligand_poses', help='either cartesian_poses, ligand_poses, '
                                                                               'or conformer_poses')
    args = parser.parse_args()

    if not os.path.exists(args.run_path):
        os.mkdir(args.run_path)

    if args.task == 'all':
        process = get_prots(args.docked_prot_file)
        grouped_files = group_files(args.n, process)
        run_all(args.docked_prot_file, args.run_path, args.raw_root, args.out_dir, grouped_files, args.n,
                args.decoy_type)

    if args.task == 'group':
        process = get_prots(args.docked_prot_file)
        grouped_files = group_files(args.n, process)
        run_group(grouped_files, args.raw_root, args.index, args.rmsd_cutoff, args.decoy_type)

    if args.task == 'check':
        run_check(args.docked_prot_file, args.raw_root, args.decoy_type)

    if args.task == 'combine':
        run_combine(args.docked_prot_file, args.raw_root, args.out_dir, args.decoy_type)

    if args.task == 'update':
        update(args.docked_prot_file, args.raw_root, args.new_prot_file)

    if args.task == 'remove':
        with open(args.docked_prot_file) as fp:
            for line in tqdm(fp, desc='files'):
                if line[0] == '#': continue
                protein, target, start = line.strip().split()
                protein_path = os.path.join(args.raw_root, protein)
                pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
                file = os.path.join(pair_path, '{}-to-{}_{}.csv'.format(target, start, args.decoy_type))
                if os.path.exists(file):
                    os.remove(file)

    if args.task == 'check_ordering':
        process = get_prots(args.docked_prot_file)
        # grouped_files = group_files(args.n, process)
        error = []
        for protein, target, start in tqdm(process, desc='protein, target, start groups'):
            protein_path = os.path.join(args.raw_root, protein)
            pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
            file = os.path.join(pair_path, '{}-to-{}.csv'.format(target, start))
            label_df = pd.read_csv(file)
            prev_pdb_code = '{}_lig1'.format(target)
            for i in range(2, args.max_poses):
                pdb_code = '{}_lig{}'.format(target, i)
                if len(label_df[label_df['target'] == pdb_code]) != 0 \
                        and get_glide_score(prev_pdb_code, label_df) > get_glide_score(pdb_code, label_df):
                    error.append((protein, target, start))
                    print(prev_pdb_code, pdb_code)
                    break
                prev_pdb_code = pdb_code

        print(len(error))
        print(error)

if __name__ == "__main__":
    main()



