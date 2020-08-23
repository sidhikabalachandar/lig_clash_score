"""
The purpose of this code is to set the train, val, and test data sets
It can be run on sherlock using
/home/groups/rondror/software/sidhikab/miniconda/envs/geometric/bin/python check_alignment.py
"""

import pickle
import numpy as np
from tqdm import tqdm
import statistics

NUM_ST_DEV = 2.5

prot_file = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random_with_unaligned.txt'
save_path = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt'
data_root = '/oak/stanford/groups/rondror/projects/ligand-docking/pdbbind_2019/data'
com_dict = '/home/users/sidhikab//flexibility_project/atom3d/src/data/com.pkl'


def get_volume(list):
    x_dim = max(list[:, 0]) - min(list[:, 0])
    y_dim = max(list[:, 1]) - min(list[:, 1])
    z_dim = max(list[:, 2]) - min(list[:, 2])
    return x_dim * y_dim * z_dim

def main():
    infile = open(com_dict, 'rb')
    center_of_masses = pickle.load(infile)
    infile.close()

    unaligned = []
    for protein in center_of_masses:
        pts = np.zeros((len(center_of_masses[protein]), 3))
        for i, ligand in enumerate(center_of_masses[protein]):
            pts[i] = center_of_masses[protein][ligand]

        avg = np.average(pts, axis=0)
        dist_list = []
        for ligand in center_of_masses[protein]:
            dist_list.append(np.linalg.norm(center_of_masses[protein][ligand] - avg))

        if len(dist_list) > 1:
            mean = statistics.mean(dist_list)
            stdev = statistics.stdev(dist_list)
            for i, ligand in enumerate(center_of_masses[protein]):
                if not mean - NUM_ST_DEV * stdev < dist_list[i] < mean + NUM_ST_DEV * stdev:
                    unaligned.append((protein, ligand))

    text = []
    to_remove = []
    with open(prot_file) as fp:
        for line in tqdm(fp, desc='protein file'):
            if line[0] == '#': continue
            protein, target, start = line.strip().split()
            if (protein, start) not in unaligned and (protein, target) not in unaligned:
                text.append(line)
            else:
                to_remove.append((protein, target, start))

    print(len(to_remove))
    print(to_remove)

    # file = open(save_path, "w")
    # file.writelines(text)
    # file.close()

if __name__=="__main__":
    main()