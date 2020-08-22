"""
The purpose of this code is to set the train, val, and test data sets
It can be run on sherlock using
ml load chemistry
ml load schrodinger
$ $SCHRODINGER/run python3 get_pocket_com.py
"""

import os
from docking.docking_class import Docking_Set
from docking.utilities import score_no_vdW
import time
import sys
from tqdm import tqdm

test_data_directory = os.path.dirname(os.path.realpath(__file__))+'/test_data'
protein_file = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt'
raw_dir = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw'

def main():
    test_directory = os.getcwd() + '/testrun3'

    # Note: docking method is set to inplace
    docking_config = [{'folder': test_directory + '/test_docking1',
                       'name': 'test_docking1',
                       'grid_file': test_data_directory + '/2B7A.zip',
                       'prepped_ligand_file': test_data_directory + '/2W1I_3_poses.mae',
                       'glide_settings': {'num_poses': 1, 'docking_method': 'inplace'}},
                      ]
    run_config = {'run_folder': test_directory + '/run',
                  'group_size': 5,
                  'partition': 'rondror',
                  'dry_run': False}

    dock_set = Docking_Set()
    dock_set.run_docking_set(docking_config, run_config)

    for i in range(1, 15):
        done_list, log_list = dock_set.check_docking_set_done(docking_config)
        if (all(done_list)):
            print("Docking Completed")

            # Note: get the scores, not that pose1 has purposeful clashes
            results = dock_set.get_docking_gscores(docking_config, mode='multi')
            results_by_ligand = results['test_docking1']
            results_by_ligand['2W1I_pose2'][0]['GScore'], -7.07
            results_by_ligand['2W1I_pose1'][0]['GScore'], 10000.00
            results_by_ligand['2W1I_pose1'][0]['vdW'], 14374956.0
            # compute the score without vdW terms
            score_no_vdW(results_by_ligand['2W1I_pose1'][0]) - 4.89 < 0.0001

            return
        else:
            print("Waiting for docking completion ...")
        time.sleep(60)

def main():
    max_ligands = 25
    combind_root = '/oak/stanford/groups/rondror/projects/combind/bpp_data'
    dock_set = Docking_Set()

    task = sys.argv[1]
    if task == 'run_dock':
        with open(protein_file) as fp:
            for line in tqdm(fp, desc='files'):
                if line[0] == '#': continue
                protein, target, start = line.strip().split()
                for file in os.listdir(os.path.join(raw_dir, '{}/{}-to-{}'.format(protein, target, start))):
                    if 'lig' in file and file[:-3] == 'mae':
                        root = '/oak/stanford/groups/rondror/projects/combind/flexibility/' + protein + '/' + protein + '_' + classifier + '/'
                        output_folder = root + 'mut_rmsds'
                        grid_folder = root + 'grids/'
                        docking_config = get_docking_info(combind_root, protein, max_ligands, output_folder, grid_folder)
                        run_config = {'run_folder': output_folder + '/run',
                                      'group_size': 15,
                                      'partition': 'rondror',
                                      'dry_run': False}
                        print(protein, classifier)
                        dock_set.run_docking_rmsd_delete(docking_config, run_config, incomplete_only=False)

    if task == 'check':
        for protein in PROTS:
            for classifier in labels:
                root = '/oak/stanford/groups/rondror/projects/combind/flexibility/' + protein + '/' + protein + '_' + classifier + '/'
                output_folder = root + 'mut_rmsds'
                grid_folder = root + '/grids'
                docking_config = get_docking_info(combind_root, protein, max_ligands, output_folder, grid_folder)
                done = dock_set.check_rmsd_set_done(docking_config)
                missing = [item[0]['name'] for item in zip(docking_config, done) if not item[1]]
                print('{}: Missing {}/{}'.format(protein, len(missing), len(docking_config)))
                print(missing)

    if task == 'results':
        rmsds = {}
        for protein in PROTS:
            for classifier in labels:
                root = '/oak/stanford/groups/rondror/projects/combind/flexibility/' + protein + '/' + protein + '_' + classifier + '/'
                output_folder = root + 'mut_rmsds'
                grid_folder = root + '/grids'
                docking_config = get_docking_info(combind_root, protein, max_ligands, output_folder, grid_folder)
                print(len(docking_config))
                done = dock_set.check_rmsd_set_done(docking_config)
                missing = [item[0]['name'] for item in zip(docking_config, done) if not item[1]]
                docking_config = get_docking_info(combind_root, protein, max_ligands, output_folder, grid_folder, missing)
                print(len(docking_config))
                docking_results = dock_set.get_docking_results(docking_config)
                struc_dict = {}
                for name, rmsd in docking_results.items():
                    # name is ligand_to_struc
                    ls = name.split('_to_')
                    if ls[1] not in struc_dict:
                        struc_dict[ls[1]] = {}
                    struc_dict[ls[1]][ls[0]] = rmsd
                rmsds[protein] = struc_dict
                with open(output_folder + '/rmsds.pkl', 'wb') as outfile:
                    pickle.dump(rmsds, outfile)

if __name__=="__main__":
    main()