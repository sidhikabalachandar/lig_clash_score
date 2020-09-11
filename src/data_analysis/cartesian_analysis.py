"""
The purpose of this code is to create the rmsd csv files, combine them into one csv file, and search in this csv file

It can be run on sherlock using
ml load chemistry
ml load schrodinger
$ $SCHRODINGER/run python3 cartesian_analysis.py /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
"""

import argparse
import os
import schrodinger.structure as structure
from schrodinger.structutils.transform import get_centroid

import sys
sys.path.append('/home/users/sidhikab/docking')
from docking.docking_class import Docking_Set
from docking.utilities import score_no_vdW

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    args = parser.parse_args()

    docking_config = []
    scores = []

    with open(args.docked_prot_file) as fp:
        for line in fp:
            if line[0] == '#': continue
            protein, target, start = line.strip().split()
            protein_path = os.path.join(args.raw_root, protein)
            pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
            pose_path = os.path.join(pair_path, 'cartesian_ligand_poses')
            docking_config.append({'folder': pair_path,
                                   'name': '{}-to-{}_cartesian'.format(target, start),
                                   'grid_file': os.path.join(pair_path, '{}-to-{}.zip'.format(target, start)),
                                   'prepped_ligand_file':
                                       os.path.join(pair_path, '{}-to-{}_cartesian_merge_pv.mae'.format(target, start)),
                                   'glide_settings': {'num_poses': 1, 'docking_method': 'inplace'}})

            dock_set = Docking_Set()
            results = dock_set.get_docking_gscores(docking_config, mode='multi')
            results_by_ligand = results['{}-to-{}_cartesian'.format(target, start)]
            for file in results_by_ligand:
                s = list(structure.StructureReader(os.path.join(pose_path, file)))[0]
                scores.append((file, get_centroid(s), score_no_vdW(results_by_ligand[file][0])))
                print(scores)
            break

if __name__ == "__main__":
    main()