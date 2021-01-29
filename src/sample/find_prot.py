"""
The purpose of this code is to get the physics scores and the rmsds

It can be run on sherlock using
$ $SCHRODINGER/run python3 find_prot.py glide_rmsd /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw --max_num_concurrent_jobs 1
"""

import argparse
import os
from tqdm import tqdm
import schrodinger.structure as structure
import schrodinger.structutils.interactions.steric_clash as steric_clash

import sys
sys.path.append('/home/users/sidhikab/docking')
import pandas as pd
import random

def get_prots(docked_prot_file):
    """
    gets list of all protein, target ligands, and starting ligands in the index file
    :param docked_prot_file: (string) file listing proteins to process
    :return: process (list) list of all protein, target ligands, and starting ligands to process
    """
    process = []
    with open(docked_prot_file) as fp:
        for line in tqdm(fp, desc='protein, target, start groups'):
            if line[0] == '#': continue
            protein, target, start = line.strip().split()
            process.append((protein, target, start))
    return process


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either run, check, or delete_json')
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    args = parser.parse_args()

    random.seed(0)
    num_found = 0

    if args.task == 'glide_rmsd':
        process = get_prots(args.docked_prot_file)
        random.shuffle(process)
        for protein, target, start in process:
            pair = '{}-to-{}'.format(target, start)
            protein_path = os.path.join(args.raw_root, protein)
            pair_path = os.path.join(protein_path, pair)
            rmsd_df = pd.read_csv(os.path.join(pair_path, '{}_rmsd.csv'.format(pair)))
            rmsds = []

            for i in range(1, 100):
                pose_df = rmsd_df[rmsd_df['Title'] == '{}_lig{}.mae'.format(target, i)]
                if len(pose_df) > 0:
                    rmsds.append((pose_df['Title'].iloc[0], pose_df['RMSD'].iloc[0]))

            if len([i for i in rmsds if i[1] < 2]) == 0:
                print(protein, target, start)
                num_found += 1
                if num_found <= 10:
                    prot_file = os.path.join(pair_path, '{}_prot.mae'.format(start))
                    pose_path = os.path.join(pair_path, 'ligand_poses')
                    target_lig_file = os.path.join(pose_path, '{}_lig0.mae'.format(target))
                    prot = list(structure.StructureReader(prot_file))[0]
                    target_lig = list(structure.StructureReader(target_lig_file))[0]
                    print(steric_clash.clash_volume(prot, struc2=target_lig))

                    with structure.StructureWriter(os.path.join(pair_path, 'all_glide_poses.mae')) as glide:
                        for i in range(1, 100):
                            file = os.path.join(pose_path, '{}_lig{}.mae'.format(target, i))
                            if os.path.exists(file):
                                glide.append(list(structure.StructureReader(file))[0])
                else:
                    break


if __name__=="__main__":
    main()