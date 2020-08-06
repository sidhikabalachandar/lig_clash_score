"""
The purpose of this code is to set the train, val, and test data sets
It can be run on sherlock using
ml load chemistry
ml load schrodinger
$ $SCHRODINGER/run python3 decoy_creator.py all
$ $SCHRODINGER/run python3 decoy_creator.py group <index>
"""

import os
import sys
import schrodinger.structure as structure
import numpy as np

prot_file = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt'
save_root = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw'
run_path = '/home/users/sidhikab/flexibility_project/atom3d/src/data/run'
MAX_POSES = 100
MAX_DECOYS = 10
MEAN_TRANSLATION = 6
STDEV_TRANSLATION = 4

def rotate(origin, point):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy, oz = origin
    px, py, pz = point

    xy_angle = np.random.uniform(0, np.pi * 2)
    px = ox + np.cos(xy_angle) * (px - ox) - np.sin(xy_angle) * (py - oy)
    py = oy + np.sin(xy_angle) * (px - ox) + np.cos(xy_angle) * (py - oy)

    xz_angle = np.random.uniform(0, np.pi * 2)
    px = ox + np.cos(xz_angle) * (px - ox) - np.sin(xz_angle) * (pz - oz)
    pz = oz + np.sin(xz_angle) * (px - ox) + np.cos(xz_angle) * (pz - oz)

    yz_angle = np.random.uniform(0, np.pi * 2)
    py = oy + np.cos(yz_angle) * (py - oy) - np.sin(yz_angle) * (pz - oz)
    pz = oz + np.sin(yz_angle) * (py - oy) + np.cos(yz_angle) * (pz - oz)
    return px, py, pz

def random_three_vector():
    """
    Generates a random 3D unit vector (direction) with a uniform spherical distribution
    Algo from http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
    :return:
    """
    phi = np.random.uniform(0,np.pi*2)
    costheta = np.random.uniform(-1,1)

    theta = np.arccos(costheta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z

def create_decoys(lig_file):
    s = list(structure.StructureReader(lig_file))[0]
    for i in range(MAX_DECOYS):
        #translation
        x, y, z = random_three_vector()
        dist = np.random.normal(MEAN_TRANSLATION, STDEV_TRANSLATION)
        coords = s.getXYZ()
        vec = np.array([x * dist, y * dist, z * dist])
        coords += vec

        #rotation
        centroid = np.mean(coords, axis=0)
        for j, coord in enumerate(coords):
            coords[j] = rotate(centroid, coord)

        s.setXYZ(coords)
        with structure.StructureWriter(lig_file[:-4] + chr(ord('a')+i) + '.mae') as decoy:
            decoy.append(s)

def main():
    task = sys.argv[1]
    process = []
    counter = 0
    with open(prot_file) as fp:
        for line in fp:
            if line[0] == '#': continue
            counter += 1
            protein, target, start = line.strip().split()
            pv_file = os.path.join(save_root,
                                   '{}/{}-to-{}/{}-to-{}_pv.maegz'.format(protein, target, start, target, start))
            if os.path.exists(pv_file):
                process.append(pv_file)

    grouped_files = []
    n = 20

    for i in range(0, len(process), n):
        grouped_files += [process[i: i + n]]

    if not os.path.exists(run_path):
        os.mkdir(run_path)

    if task == 'all':
        for i, group in enumerate(grouped_files):
            cmd = 'sbatch -p owners -t 1:00:00 -o {} --wrap="$SCHRODINGER/run python3 decoy_creator.py group {}"'
            os.system(cmd.format(os.path.join(run_path, 'decoy{}.out'.format(i)), i))
            # print(cmd.format(os.path.join(run_path, 'decoy{}.out'.format(i)), i))

    if task == 'group':
        i = int(sys.argv[2])

        for file in grouped_files[i]:
            protein = file.split('/')[-3]
            [target, start] = file.split('/')[-2].split('-to-')
            num_poses = len(list(structure.StructureReader(file)))
            for i in range(num_poses):
                if i == 101:
                    break
                lig_file = '{}/{}/{}-to-{}/{}_lig{}.mae'.format(save_root, protein, target, start, target, i)
                print(lig_file)
                create_decoys(lig_file)

if __name__=="__main__":
    main()