

"""
The purpose of this code is to create conformers

It can be run on sherlock using
$ $SCHRODINGER/run python3 find_best.py /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
"""

import argparse
import os
import pandas as pd
import schrodinger.structure as structure
from schrodinger.structutils.transform import get_centroid
import numpy as np


def get_grid_size(pair_path, target, start):
    target_lig_file = os.path.join(pair_path, 'ligand_poses', '{}_lig0.mae'.format(target))
    target_lig = list(structure.StructureReader(target_lig_file))[0]
    target_center = get_centroid(target_lig)

    start_lig_file = os.path.join(pair_path, '{}_lig.mae'.format(start))
    start_lig = list(structure.StructureReader(start_lig_file))[0]
    start_center = get_centroid(start_lig)

    dist = np.sqrt((target_center[0] - start_center[0]) ** 2 +
                   (target_center[1] - start_center[1]) ** 2 +
                   (target_center[2] - start_center[2]) ** 2)

    grid_size = int(dist + 1)
    if grid_size % 2 == 1:
        grid_size += 1
    return grid_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_root', type=str, help='directory where script and output files will be written')
    args = parser.parse_args()

    protein, target, start = 'P11838', '3wz6', '1gvx'
    print(protein, target, start)
    pair = '{}-to-{}'.format(target, start)
    protein_path = os.path.join(args.raw_root, protein)
    pair_path = os.path.join(protein_path, pair)

    grid_size = get_grid_size(pair_path, target, start)
    group_name = 'exhaustive_grid_{}_2_rotation_0_360_20_rmsd_2.5'.format(grid_size)
    pose_path = os.path.join(pair_path, group_name)
    overall_min_rmsd = 1000
    min_name = ''
    min_file = ''
    for file in os.listdir(pose_path):
        prefix = 'exhaustive_search_poses'
        if file[:len(prefix)] == prefix:
            df = pd.read_csv(os.path.join(pose_path, file))
            min_rmsd = df['rmsd'].min()
            if min_rmsd < overall_min_rmsd:
                overall_min_rmsd = min_rmsd
                min_name = df[df['rmsd'] == overall_min_rmsd]['name'].iloc[0]
                min_file = file
                print(min_rmsd, min_name, min_file)

    print('OVERALL MIN')
    print(min_rmsd, min_name, min_file)


if __name__ == "__main__":
    main()

