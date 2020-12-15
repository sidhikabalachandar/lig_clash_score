"""
The purpose of this code is to first create the raw directory folder and include the following files
starting protein receptor
starting ligand
target ligand
glide pose viewer file

Then the top glide poses are added

Then the decoys are created

It can be run on sherlock using
$ $SCHRODINGER/run python3 decoy_efficacy.py indv /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /home/users/sidhikab/lig_clash_score/src/data/decoys_all --protein Q16539 --target 2bak --start 3d7z --index 43
"""

import argparse
import os
import schrodinger.structure as structure
import schrodinger.structutils.transform as transform
from schrodinger.structutils.transform import get_centroid
import schrodinger.structutils.interactions.steric_clash as steric_clash
import numpy as np
import schrodinger.structutils.rmsd as rmsd
import random
import subprocess
from tqdm import tqdm
import math


def get_jobs_in_queue(code):
    cmd = ['squeue', '-u', os.environ['USER']]
    slurm = subprocess.run(cmd, stdout=subprocess.PIPE, encoding='utf-8')
    count = 0
    for ln in slurm.stdout.split('\n'):
        terms = ln.split()
        if len(terms) > 2 and terms[2].strip()[:3] == code:
            count += 1
    return count


def create_conformer_decoys(save_path, run_path, conformers, grid, num_jobs_submitted, start_lig_center, target_lig,
                            prot, min_angle, max_angle, rmsd_cutoff, protein, target, start, index):
    conformer_ls = [[c, 0] for c in conformers]

    rot_ls = []
    for rot_x in range(int(math.degrees(min_angle)), int(math.degrees(max_angle)) + 1):
        for rot_y in range(int(math.degrees(min_angle)), int(math.degrees(max_angle)) + 1):
            for rot_z in range(int(math.degrees(min_angle)), int(math.degrees(max_angle)) + 1):
                rot_ls.append([[math.radians(rot_x), math.radians(rot_y), math.radians(rot_z)], 0])

    output_file = os.path.join(run_path, '{}_{}_{}_{}.txt'.format(protein, target, start, index))
    num_iter_without_pose = 0
    num_valid_poses = 0
    num_total_poses = 0

    while True:
        num_iter_without_pose += 1
        num_total_poses += 1
        if num_total_poses % 1000 == 0:
            num_jobs_in_queue = get_jobs_in_queue('{}{}{}'.format(protein[0], target[0], start[0]))
            f = open(output_file, "a")
            f.write("num_total_poses: {}, len(grid): {}, len(conformer_ls): {}, len(rot_ls): {}, num_jobs_in_queue: "
                    "{}\n".format(num_total_poses, len(grid), len(conformer_ls), len(rot_ls), num_jobs_in_queue))
            f.close()
            if num_jobs_in_queue != num_jobs_submitted:
                break
        conformer_index = random.randint(0, len(conformer_ls) - 1)
        conformer = conformer_ls[conformer_index][0]
        conformer_center = list(get_centroid(conformer))

        # translation
        index = random.randint(0, len(grid) - 1)
        grid_loc = grid[index][0]
        transform.translate_structure(conformer, start_lig_center[0] - conformer_center[0] + grid_loc[0],
                                      start_lig_center[1] - conformer_center[1] + grid_loc[1],
                                      start_lig_center[2] - conformer_center[2] + grid_loc[2])
        conformer_center = list(get_centroid(conformer))

        # rotation
        if len(grid) > 1:
            x_angle = np.random.uniform(min_angle, max_angle)
            y_angle = np.random.uniform(min_angle, max_angle)
            z_angle = np.random.uniform(min_angle, max_angle)
        else:
            rot_index = random.randint(0, len(rot_ls) - 1)
            x_angle, y_angle, z_angle = rot_ls[rot_index][0]
        transform.rotate_structure(conformer, x_angle, y_angle, z_angle, conformer_center)

        if steric_clash.clash_volume(prot, struc2=conformer) < 200:
            num_valid_poses += 1
            if rmsd.calculate_in_place_rmsd(conformer, conformer.getAtomIndices(), target_lig,
                                            target_lig.getAtomIndices()) < rmsd_cutoff:
                save_file = os.path.join(save_path, '{}_{}_{}.txt'.format(protein, target, start))
                f = open(output_file, "a")
                f.write("Num poses searched = {}\n".format(num_total_poses))
                f.write("Num acceptable clash poses searched = {}\n".format(num_valid_poses))
                f.close()
                if not os.path.exists(save_file):
                    with open(save_file, 'w') as f:
                        f.write("Num poses searched = {}\n".format(num_total_poses))
                        f.write("Num acceptable clash poses searched = {}\n".format(num_valid_poses))
                break
            grid[index][1] = 0
            num_iter_without_pose = 0
        elif num_iter_without_pose == 5 and len(grid) > 1:
            max_val = max(grid, key=lambda x: x[1])
            grid.remove(max_val)
            num_iter_without_pose = 0
        elif num_iter_without_pose == 5 and len(grid) == 1:
            if len(conformer_ls) == 1 and len(rot_ls) == 1:
                save_file = os.path.join(save_path, '{}_{}_{}.txt'.format(protein, target, start))
                f = open(output_file, "a")
                f.write("Num poses searched = {}\n".format(num_total_poses))
                f.write("Num acceptable clash poses searched = {}\n".format(num_valid_poses))
                f.write("No correct poses found\n")
                f.close()
                if not os.path.exists(save_file):
                    with open(save_file, 'w') as f:
                        f.write("Num poses searched = {}\n".format(num_total_poses))
                        f.write("Num acceptable clash poses searched = {}\n".format(num_valid_poses))
                        f.write("No correct poses found\n")
                break
            elif len(conformer_ls) > 1 and (len(rot_ls) == 1 or (len(conformer_ls) + len(rot_ls)) % 2 == 0):
                max_val = max(conformer_ls, key=lambda x: x[1])
                conformer_ls.remove(max_val)
            else:
                max_val = max(rot_ls, key=lambda x: x[1])
                rot_ls.remove(max_val)
            num_iter_without_pose = 0
        else:
            grid[index][1] += 1
            conformer_ls[conformer_index][1] += 1
            if len(grid) == 1:
                rot_ls[rot_index][1] += 1


def run_group(protein, target, start, raw_root, save_path, run_path, min_angle, max_angle, index, rmsd_cutoff, grid,
              num_jobs_submitted):
    """
    creates decoys for each protein, target, start group
    :param grouped_files: (list) list of protein, target, start groups
    :param raw_root: (string) directory where raw data will be placed
    :param data_root: (string) pdbbind directory where raw data will be obtained
    :param index: (int) group number
    :param max_poses: (int) maximum number of glide poses considered
    :param decoy_type: (string) either cartesian or random
    :param max_decoys: (int) maximum number of decoys created per glide pose
    :param mean_translation: (float) mean distance decoys are translated
    :param stdev_translation: (float) stdev of distance decoys are translated
    :param min_angle: (float) minimum angle decoys are rotated
    :param max_angle: (float) maximum angle decoys are rotated
    :return:
    """
    pair = '{}-to-{}'.format(target, start)
    protein_path = os.path.join(raw_root, protein)
    pair_path = os.path.join(protein_path, pair)
    start_lig_file = os.path.join(pair_path, '{}_lig.mae'.format(start))
    start_lig = list(structure.StructureReader(start_lig_file))[0]
    target_lig_file = os.path.join(pair_path, 'ligand_poses', '{}_lig0.mae'.format(target))
    target_lig = list(structure.StructureReader(target_lig_file))[0]
    start_lig_center = list(get_centroid(start_lig))
    prot_file = os.path.join(pair_path, '{}_prot.mae'.format(start))
    prot = list(structure.StructureReader(prot_file))[0]

    aligned_file = os.path.join(pair_path, "aligned_conformers.mae")
    conformers = list(structure.StructureReader(aligned_file))

    create_conformer_decoys(save_path, run_path, conformers, grid, num_jobs_submitted, start_lig_center, target_lig, prot,
                            min_angle, max_angle, rmsd_cutoff, protein, target, start, index)

def get_prots(docked_prot_file):
    """
    gets list of all protein, target ligands, and starting ligands in the index file
    :param docked_prot_file: (string) file listing proteins to process
    :return: process (list) list of all protein, target ligands, and starting ligands to process
    """
    process = []
    with open(docked_prot_file) as fp:
        for line in tqdm(fp, desc='index file'):
            if line[0] == '#': continue
            protein, target, start = line.strip().split()
            process.append((protein, target, start))

    return process

def get_grid_groups(grid_size, n):
    grid = []
    for dx in range(-grid_size, grid_size):
        for dy in range(-grid_size, grid_size):
            for dz in range(-grid_size, grid_size):
                grid.append([[dx, dy, dz], 0])

    grouped_files = []

    for i in range(0, len(grid), n):
        grouped_files += [grid[i: i + n]]

    return grouped_files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either all, group, check, '
                                               'all_dist_check, group_dist_check, check_dist_check, '
                                               'all_name_check, group_name_check, check_name_check, or delete')
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('save_path', type=str, help='directory where results will be saved')
    parser.add_argument('--min_angle', type=float, default=- np.pi / 6, help='minimum angle decoys are rotated')
    parser.add_argument('--max_angle', type=float, default=np.pi / 6, help='maximum angle decoys are rotated')
    parser.add_argument('--grid_size', type=int, default=6, help='grid size in positive and negative x, y, z '
                                                                 'directions')
    parser.add_argument('--index', type=int, default=-1, help='grid point group index')
    parser.add_argument('--rmsd_cutoff', type=int, default=2, help='rmsd accuracy cutoff between predicted ligand pose '
                                                                   'and true ligand pose')
    parser.add_argument('--protein', type=str, default='', help='protein name')
    parser.add_argument('--target', type=str, default='', help='target ligand name')
    parser.add_argument('--start', type=str, default='', help='start ligand name')
    parser.add_argument('--n', type=int, default=20, help='number of grid points processed in indv task')
    args = parser.parse_args()

    if not os.path.exists(args.run_path):
        os.mkdir(args.run_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    random.seed(0)

    if args.task == 'all':
        grouped_files = get_grid_groups(args.grid_size, args.n)
        process = get_prots(args.docked_prot_file)
        random.shuffle(process)
        submitted_codes = []
        i = 0
        while len(submitted_codes) < 1:
            protein, target, start = process[i]
            i += 1
            code = '{}{}{}'.format(protein[0], target[0], start[0])
            # if code not in submitted_codes:
            if code == 'Q23':
                submitted_codes.append(code)
                if not os.path.exists(os.path.join(args.save_path, '{}_{}_{}.txt'.format(protein, target, start))):
                    for j in range(len(grouped_files)):
                        z_code = '{}{}'.format(code, j)
                        os.system('sbatch -p owners -t 1:00:00 -o {} -J {} --wrap="$SCHRODINGER/run python3 '
                                  'decoy_efficacy.py indv {} {} {} {} --index {} --protein {} --target {} --start '
                                  '{}"'.format(os.path.join(args.run_path, '{}.out'.format(z_code)), z_code,
                                               args.docked_prot_file, args.run_path, args.raw_root, args.save_path,
                                               j, protein, target, start))
        print(i)
        print(submitted_codes)

    elif args.task == 'indv':
        grouped_files = get_grid_groups(args.grid_size, args.n)
        run_group(args.protein, args.target, args.start, args.raw_root, args.save_path, args.run_path, args.min_angle,
                  args.max_angle, args.index, args.rmsd_cutoff, grouped_files[args.index], len(grouped_files))

if __name__=="__main__":
    main()