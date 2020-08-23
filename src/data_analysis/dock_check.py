'''
This protocol can be used to check how many proteins in docked_prot_file are actually docked

how to run this file:
/home/groups/rondror/software/sidhikab/miniconda/envs/geometric/bin/python dock_check.py /home/users/sidhikab/plep/index/refined_random.txt
'''

import argparse
import os

data_root = '/oak/stanford/groups/rondror/projects/ligand-docking/pdbbind_2019/data'
save_path = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    args = parser.parse_args()

    error_count = 0
    text = []
    with open(args.docked_prot_file) as fp:
        for line in fp:
            if line[0] == '#': continue
            protein, target, start = line.strip().split()

            #check if docking info exists
            dock_root = os.path.join(data_root, '{}/docking/sp_es4/{}-to-{}'.format(protein, target, start))
            if not os.path.exists(dock_root):
                error_count += 1
                continue

            #check if receptor and ligand files exist
            start_receptor_file = os.path.join(data_root, '{}/structures/aligned/{}_prot.mae'.format(protein, start))
            target_receptor_file = os.path.join(data_root, '{}/structures/aligned/{}_prot.mae'.format(protein, start))
            start_ligand_file = os.path.join(data_root, '{}/structures/aligned/{}_lig.mae'.format(protein, target))
            target_ligand_file = os.path.join(data_root, '{}/structures/aligned/{}_lig.mae'.format(protein, target))
            if not os.path.exists(start_receptor_file) or not os.path.exists(target_receptor_file) or not os.path.exists(start_ligand_file) or not os.path.exists(target_ligand_file):
                error_count += 1
                continue

            #check if pose viewer file exists
            pv_file = '{}/{}-to-{}_pv.maegz'.format(dock_root, target, start)
            if not os.path.exists(pv_file):
                error_count += 1
                continue

            text.append(line)

    print(error_count, "files not found")

    file = open(save_path, "w")
    file.writelines(text)
    file.close()

if __name__ == '__main__':
    main()