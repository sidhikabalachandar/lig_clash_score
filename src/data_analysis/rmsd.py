"""
The purpose of this code is to set the train, val, and test data sets
It can be run on sherlock using
/home/groups/rondror/software/sidhikab/miniconda/envs/geometric/bin/python rmsd.py
"""

import os

prot_file = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt'
save_root = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw'
data_root = '/oak/stanford/groups/rondror/projects/ligand-docking/pdbbind_2019/data'
run_root = '/home/users/sidhikab/flexibility_project/atom3d/src/data/run'

def main():
    process = []
    counter = 0
    with open(prot_file) as fp:
        for line in fp:
            if line[0] == '#': continue
            counter += 1
            protein, target, start = line.strip().split()
            pv_file = os.path.join(save_root,
                                   '{}/{}-to-{}/{}-to-{}_pv.maegz'.format(protein, target, start, target, start))
            rmsd_file = '{}/{}/docking/sp_es4/{}-to-{}/rmsd.csv'.format(data_root, protein, target, start)
            if os.path.exists(pv_file) and not os.path.exists(rmsd_file):
                process.append((protein, target, start))

    print('Missing', len(process), '/', counter)

    grouped_files = []
    n = 5

    for i in range(0, len(process), n):
        grouped_files += [process[i: i + n]]

    if not os.path.exists(run_root):
        os.mkdir(run_root)

    for i, group in enumerate(grouped_files):
        sh_file = os.path.join(run_root, 'rmsd{}_in.sh'.format(i))
        with open(sh_file, 'w') as f:
            f.write('#!/bin/bash\n')
            for dock_set in group:
                protein, target, start = dock_set
                f.write('cd {}/{}/docking/sp_es4/{}-to-{}\n'.format(data_root, protein, target, start))
                rmsd_file_name = '{}/{}/docking/sp_es4/{}-to-{}/rmsd.csv'.format(data_root, protein, target, start)
                ligand_file_name = '{}/{}/structures/aligned/{}_lig.mae'.format(data_root, protein, target)
                pose_viewer_file_name = '{}/{}/docking/sp_es4/{}-to-{}/{}-to-{}_pv.maegz'.format(data_root, protein, target, start, target, start)
                rmsd_cmd = '$SCHRODINGER/run rmsd.py -use_neutral_scaffold -pv second -c {} {} {}\n'
                f.write(rmsd_cmd.format(rmsd_file_name, ligand_file_name, pose_viewer_file_name))
        os.chdir(run_root)
        os.system('sbatch -p rondror -t 02:00:00 -o rmsd{}.out rmsd{}_in.sh'.format(i, i))
        # print('sbatch -p rondror -t 02:00:00 -o rmsd{}.out rmsd{}_in.sh'.format(i, i))

if __name__=="__main__":
    main()

