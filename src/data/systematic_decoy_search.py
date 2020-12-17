"""
The purpose of this code is to first create the raw directory folder and include the following files
starting protein receptor
starting ligand
target ligand
glide pose viewer file

Then the top glide poses are added

Then the decoys are created

It can be run on sherlock using
$ $SCHRODINGER/run python3 systematic_decoy_search.py run_search /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw --protein A0A0J9 --target 4or4 --start 4q46 --index 0
"""

import argparse
import os
import schrodinger.structure as structure
import subprocess
import schrodinger.structutils.transform as transform
from schrodinger.structutils.transform import get_centroid
import schrodinger.structutils.interactions.steric_clash as steric_clash
import schrodinger.structutils.rmsd as rmsd
import math
import time
import statistics

_ALIGN_CMD = "$SCHRODINGER/shape_screen -shape {shape} -screen {screen} -WAIT -JOB {job_name}"

def run_cmd(cmd, error_msg=None, raise_except=False):
    try:
        return subprocess.check_output(
            cmd,
            universal_newlines=True,
            shell=True)
    except Exception as e:
        if error_msg is not None:
            print(error_msg)
        if raise_except:
            raise e

def create_conformer_decoys(conformers, start_lig_center, target_lig, prot, get_clash_rmsd_time, cutoff,
                            rotation_search_step_size):
    out_file = os.path.join(os.getcwd(), 'decoy_time_step_size_{}.out'.format(rotation_search_step_size))
    decoy_start_time = time.time()
    clash_times = []
    rmsd_times = []
    num_correct_found = 0
    counter = 0
    for conformer in conformers:
        for x_angle_deg in range(-30, 30, rotation_search_step_size):
            for y_angle_deg in range(-30, 30, rotation_search_step_size):
                for z_angle_deg in range(-30, 30, rotation_search_step_size):
                    counter += 1
                    if counter % 1000 == 0:
                        f = open(out_file, "a")
                        f.write("Num poses searched = {}, num correct poses found = {}, time elapsed = {}\n".format(
                            counter, num_correct_found, time.time() - decoy_start_time))
                        f.close()
                    x_angle_rad = math.radians(x_angle_deg)
                    y_angle_rad = math.radians(y_angle_deg)
                    z_angle_rad = math.radians(z_angle_deg)

                    conformer_center = list(get_centroid(conformer))

                    # translation
                    grid_loc = [0, 0, 0]
                    transform.translate_structure(conformer, start_lig_center[0] - conformer_center[0] + grid_loc[0],
                                                  start_lig_center[1] - conformer_center[1] + grid_loc[1],
                                                  start_lig_center[2] - conformer_center[2] + grid_loc[2])
                    conformer_center = list(get_centroid(conformer))

                    # rotation
                    transform.rotate_structure(conformer, x_angle_rad, y_angle_rad, z_angle_rad, conformer_center)

                    if get_clash_rmsd_time:
                        # get clash
                        start = time.time()
                        steric_clash.clash_volume(prot, struc2=conformer)
                        end = time.time()
                        clash_times.append(end - start)

                        # get rmsd
                        start = time.time()
                        rmsd.calculate_in_place_rmsd(conformer, conformer.getAtomIndices(), target_lig,
                                                     target_lig.getAtomIndices())
                        end = time.time()
                        rmsd_times.append(end - start)
                        if len(clash_times) == 1000:
                            print("Average clash time =", statistics.mean(clash_times))
                            print("Average rmsd time =", statistics.mean(rmsd_times))
                    else:
                        rmsd_val = rmsd.calculate_in_place_rmsd(conformer, conformer.getAtomIndices(), target_lig,
                                                                target_lig.getAtomIndices())
                        if rmsd_val < cutoff:
                            num_correct_found += 1

    decoy_end_time = time.time()
    f = open(out_file, "a")
    f.write("Total num poses searched = {}, total num correct poses found = {}, total time elapsed = {}\n".format(
        counter, num_correct_found, decoy_end_time - decoy_start_time))
    f.close()

def run_search(protein, target, start, raw_root, get_clash_rmsd_time, cutoff, rotation_search_step_size):
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

    conformer_file = os.path.join(pair_path, "aligned_to_start_conformers.mae")
    conformers = list(structure.StructureReader(conformer_file))
    create_conformer_decoys(conformers, start_lig_center, target_lig, prot, get_clash_rmsd_time, cutoff,
                            rotation_search_step_size)

def get_conformer_groups(n, target, start, protein, raw_root):
    pair = '{}-to-{}'.format(target, start)
    protein_path = os.path.join(raw_root, protein)
    pair_path = os.path.join(protein_path, pair)
    conformer_file = os.path.join(pair_path, "{}_lig0-out.maegz".format(target))
    conformers = list(structure.StructureReader(conformer_file))
    grouped_files = []

    for i in range(0, len(conformers), n):
        grouped_files.append(conformers[i: i + n])

    return grouped_files

def run_align_all(grouped_files, run_path, raw_root, n, protein, target, start):
    for i, group in enumerate(grouped_files):
        cmd = 'sbatch -p owners -t 1:00:00 -o {} --wrap="$SCHRODINGER/run python3 systematic_decoy_search.py ' \
              'align_group {} {} --n {} --index {} --protein {} --target {} --start {}"'
        os.system(cmd.format(os.path.join(run_path, 'decoy{}.out'.format(i)), run_path, raw_root, n, i, protein, target,
                             start))

def run_align_group(grouped_files, index, n, protein, target, start, raw_root):
    for i, conformer in enumerate(grouped_files[index]):
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, pair)
        output_path = os.path.join(pair_path, 'conformers')
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        num = str(n * index + i)
        os.chdir(output_path)
        screen_file = os.path.join(output_path, "screen_{}.mae".format(num))
        print(conformer)
        with structure.StructureWriter(screen_file) as screen:
            screen.append(conformer)
        shape_file = os.path.join(pair_path, '{}_lig.mae'.format(start))
        run_cmd(_ALIGN_CMD.format(shape=shape_file, screen=screen_file, job_name=num))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either align or search')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('--protein', type=str, default='', help='protein name')
    parser.add_argument('--target', type=str, default='', help='target ligand name')
    parser.add_argument('--start', type=str, default='', help='start ligand name')
    parser.add_argument('--n', type=int, default=3, help='number of alignments processed in each job')
    parser.add_argument('--rotation_search_step_size', type=int, default=1, help='step size between each angle '
                                                                                 'checked, in degrees')
    parser.add_argument('--index', type=int, default=-1, help='grid point group index')
    parser.add_argument('--rmsd_cutoff', type=int, default=2, help='rmsd accuracy cutoff between predicted ligand pose '
                                                                   'and true ligand pose')
    parser.add_argument('--time_clash_rmsd', dest='get_clash_rmsd_time', action='store_true')
    parser.add_argument('--no_time_clash_rmsd', dest='get_clash_rmsd_time', action='store_false')
    parser.set_defaults(get_clash_rmsd_time=False)

    args = parser.parse_args()

    if not os.path.exists(args.run_path):
        os.mkdir(args.run_path)

    if args.task == 'align_all':
        grouped_files = get_conformer_groups(args.n, args.target, args.start, args.protein, args.raw_root)
        run_align_all(grouped_files, args.run_path, args.raw_root, args.n, args.protein, args.target, args.start)

    elif args.task == 'align_group':
        grouped_files = get_conformer_groups(args.n, args.target, args.start, args.protein, args.raw_root)
        run_align_group(grouped_files, args.index, args.n, args.protein, args.target, args.start, args.raw_root)

    elif args.task == 'align_check':
        pair = '{}-to-{}'.format(args.target, args.start)
        protein_path = os.path.join(args.raw_root, args.protein)
        pair_path = os.path.join(protein_path, pair)
        conformer_file = os.path.join(pair_path, "{}_lig0-out.maegz".format(args.target))
        conformers = list(structure.StructureReader(conformer_file))
        output_path = os.path.join(pair_path, 'conformers')
        process = []
        for i in range(len(conformers)):
            if not os.path.exists(os.path.join(output_path, '{}_align.maegz'.format(i))):
                process.append(i)

        print("Missing: {}/{}".format(len(process), len(conformers)))
        print(process)

    elif args.task == 'align_combine':
        pair = '{}-to-{}'.format(args.target, args.start)
        protein_path = os.path.join(args.raw_root, args.protein)
        pair_path = os.path.join(protein_path, pair)
        conformer_file = os.path.join(pair_path, "{}_lig0-out.maegz".format(args.target))
        conformers = list(structure.StructureReader(conformer_file))
        output_path = os.path.join(pair_path, 'conformers')

        combined_file = os.path.join(pair_path, "aligned_to_start_conformers.mae")
        with structure.StructureWriter(combined_file) as combined:
            for i in range(len(conformers)):
                aligned_file = os.path.join(output_path, '{}_align.maegz'.format(i))
                s = list(structure.StructureReader(aligned_file))[0]
                combined.append(s)

        print(len(list(structure.StructureReader(combined_file))))

    elif args.task == 'run_search':
        cmd = 'sbatch -p owners -t 10:00:00 -o {} --wrap="$SCHRODINGER/run python3 systematic_decoy_search.py ' \
              'search {} {} --protein {} --target {} --start {} --rotation_search_step_size {}"'
        os.system(cmd.format(os.path.join(args.run_path, 'decoy_time_step_size_{}.out'.format(
            args.rotation_search_step_size)), args.run_path, args.raw_root, args.protein, args.target, args.start,
                             args.rotation_search_step_size))

    elif args.task == 'search':
        run_search(args.protein, args.target, args.start, args.raw_root, args.get_clash_rmsd_time, args.rmsd_cutoff,
                   args.rotation_search_step_size)

if __name__=="__main__":
    main()