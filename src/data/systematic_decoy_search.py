"""
The purpose of this code is to first create the raw directory folder and include the following files
starting protein receptor
starting ligand
target ligand
glide pose viewer file

Then the top glide poses are added

Then the decoys are created

It can be run on sherlock using
$ $SCHRODINGER/run python3 systematic_decoy_search.py search /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw --rotation_search_step_size 5 --time_clash_rmsd --prot_pocket_only --protein P35968 --target 4agc --start 3vhk
"""

import argparse
import os
import schrodinger.structure as structure
import subprocess
import schrodinger.structutils.transform as transform
from schrodinger.structutils.transform import get_centroid
import schrodinger.structutils.interactions.steric_clash as steric_clash
import schrodinger.structutils.rmsd as rmsd
import schrodinger.structutils.build as build
import math
import time
import statistics
import random
from tqdm import tqdm
import pandas as pd
import scipy.spatial

_CONFGEN_CMD = ("$SCHRODINGER/confgenx -WAIT -optimize -drop_problematic -num_conformers {num_conformers} "
                "-max_num_conformers {num_conformers} {input_file}")
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


def create_conformer_decoys(conformers, start_lig_center, target_lig, prot, get_clash_rmsd_time, cutoff,
                            rotation_search_step_size, protein, target, start, no_prot_h, test, x_rot, y_rot, z_rot):
    out_file_template = '{}_{}-to-{}_step_size_{}_no_lig_h.out'
    out_file = os.path.join(os.getcwd(), 'decoy_timing_data', out_file_template.format(protein, target, start,
                                                                                       rotation_search_step_size))
    decoy_start_time = time.time()
    clash_times = []
    rmsd_times = []
    num_correct_found = 0
    counter = 0
    rotation_search_step_size_rad = math.radians(rotation_search_step_size)

    for conformer in conformers:
        conformer_center = list(get_centroid(conformer))

        # translation
        grid_loc = [0, 0, 0]
        transform.translate_structure(conformer, start_lig_center[0] - conformer_center[0] + grid_loc[0],
                                      start_lig_center[1] - conformer_center[1] + grid_loc[1],
                                      start_lig_center[2] - conformer_center[2] + grid_loc[2])
        conformer_center = list(get_centroid(conformer))

        # keep track of rotation angles
        transform.rotate_structure(conformer, math.radians(-30 - rotation_search_step_size), 0, 0, conformer_center)
        x_so_far = -30 - rotation_search_step_size
        y_so_far = 0
        z_so_far = 0

        for _ in range(-30, 30, rotation_search_step_size):
            # x rotation
            transform.rotate_structure(conformer, rotation_search_step_size_rad,
                                       math.radians(-30 - rotation_search_step_size - y_so_far), 0, conformer_center)
            x_so_far += 1
            y_so_far += -30 - rotation_search_step_size - y_so_far

            for _ in range(-30, 30, rotation_search_step_size):
                # y rotation
                transform.rotate_structure(conformer, 0, rotation_search_step_size_rad,
                                           math.radians(-30 - rotation_search_step_size - z_so_far), conformer_center)
                y_so_far += 1
                z_so_far += -30 - rotation_search_step_size - z_so_far

                for _ in range(-30, 30, rotation_search_step_size):
                    # z rotation
                    transform.rotate_structure(conformer, 0, 0, rotation_search_step_size_rad, conformer_center)
                    z_so_far += 1
                    counter += 1

                    if test and x_rot == x_so_far and y_rot == y_so_far and z_rot == z_so_far:
                        return conformer

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
                            return None
                    else:
                        if counter % 1000 == 0:
                            f = open(out_file, "a")
                            f.write("Num poses searched = {}, num correct poses found = {}, time elapsed = {}\n".format(
                                counter, num_correct_found, time.time() - decoy_start_time))
                            f.close()
                        rmsd_val = rmsd.calculate_in_place_rmsd(conformer, conformer.getAtomIndices(),
                                                                target_lig, target_lig.getAtomIndices())
                        if rmsd_val < cutoff:
                            num_correct_found += 1

    decoy_end_time = time.time()
    f = open(out_file, "a")
    f.write("Total num poses searched = {}, total num correct poses found = {}, total time elapsed = {}\n".format(
        counter, num_correct_found, decoy_end_time - decoy_start_time))
    f.close()
    return None


def run_conformer_all(process, raw_root, run_path, docked_prot_file):
    counter = 0
    for protein, target, start in process:
        if counter == 10:
            break
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, pair)
        conformer_file = os.path.join(pair_path, "{}_lig0-out.maegz".format(target))
        if not os.path.exists(conformer_file):
            cmd = 'sbatch -p owners -t 1:00:00 -o {} --wrap="$SCHRODINGER/run python3 systematic_decoy_search.py ' \
                  'conformer_group {} {} {} --protein {} --target {} --start {}"'
            os.system(cmd.format(os.path.join(run_path, 'conformer_{}_{}-to-{}.out'.format(protein, target, start)),
                                 docked_prot_file, run_path, raw_root, protein, target, start))
            counter += 1
        else:
            conformers = list(structure.StructureReader(conformer_file))
            if len(conformers) > 1:
                counter += 1
            print("found:", protein, target, start)


def gen_ligand_conformers(path, output_dir, num_conformers):
    current_dir = os.getcwd()
    os.chdir(output_dir)
    basename = os.path.basename(path)
    ### Note: For some reason, confgen isn't able to find the .mae file,
    # unless it is in working directory. So, we need to copy it over.
    ### Note: There may be duplicated ligand names (for different targets).
    # Since it only happens for CHEMBL ligand, just ignore it for now.
    # Otherwise, might want consider to generate the conformers to separate
    # folders for each (target, ligand) pair.
    # Run ConfGen
    run_cmd(f'cp {path:} ./{basename:}')
    command = _CONFGEN_CMD.format(num_conformers=num_conformers,
                                  input_file=f'./{basename:}')
    run_cmd(command, f'Failed to run ConfGen on {path:}')
    run_cmd(f'rm ./{basename:}')
    os.chdir(current_dir)


def run_conformer_check(process, raw_root):
    unfinished = []
    counter = 0
    for protein, target, start in process:
        if counter == 10:
            break
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, pair)
        if not os.path.exists(os.path.join(pair_path, '{}_lig0-out.maegz'.format(target))):
            process.append((protein, target, start))
        else:
            conformer_file = os.path.join(pair_path, "{}_lig0-out.maegz".format(target))
            conformers = list(structure.StructureReader(conformer_file))
            print(protein, target, start, len(conformers), counter)
            if len(conformers) > 1:
                counter += 1

    print("Missing:", len(unfinished), "/ 10")
    print(unfinished)


def run_align_all(process, raw_root, run_path, docked_prot_file, n):
    counter = 0
    for protein, target, start in process:
        if counter == 10:
            break
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, pair)

        conformer_file = os.path.join(pair_path, "{}_lig0-out.maegz".format(target))
        conformers = list(structure.StructureReader(conformer_file))
        if len(conformers) == 1:
            continue
        else:
            counter += 1

        output_path = os.path.join(pair_path, 'conformers')
        if not os.path.exists(os.path.join(output_path, '0_align_without_hydrogen.mae')):
            print(protein, target, start, counter)
            grouped_files = get_conformer_groups(n, target, start, protein, raw_root)
            for i, group in enumerate(grouped_files):
                cmd = 'sbatch -p owners -t 1:00:00 -o {} --wrap="$SCHRODINGER/run python3 systematic_decoy_search.py ' \
                      'align_group {} {} {} --n {} --index {} --protein {} --target {} --start {}"'
                os.system(cmd.format(os.path.join(run_path, 'decoy{}.out'.format(i)), docked_prot_file, run_path,
                                     raw_root, n, i, protein, target, start))


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
        with structure.StructureWriter(screen_file) as screen:
            screen.append(conformer)
        shape_file = os.path.join(pair_path, '{}_lig.mae'.format(start))
        run_cmd(_ALIGN_CMD.format(shape=shape_file, screen=screen_file, job_name=num))
        aligned_conformer_file = os.path.join(output_path, '{}_align.maegz'.format(num))
        aligned_conformer = list(structure.StructureReader(aligned_conformer_file))[0]
        build.delete_hydrogens(aligned_conformer)
        no_hydrogen_file = os.path.join(output_path, "{}_align_without_hydrogen.mae".format(num))
        with structure.StructureWriter(no_hydrogen_file) as no_h:
            no_h.append(aligned_conformer)


def run_align_check(process, raw_root):
    unfinished = []
    counter = 0
    for protein, target, start in process:
        if counter == 10:
            break
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, pair)

        conformer_file = os.path.join(pair_path, "{}_lig0-out.maegz".format(target))
        conformers = list(structure.StructureReader(conformer_file))
        print(protein, target, start, counter)
        if len(conformers) == 1:
            continue
        else:
            counter += 1
        output_path = os.path.join(pair_path, 'conformers')

        for i in range(len(conformers)):
            if not os.path.exists(os.path.join(output_path, '{}_align_without_hydrogen.mae'.format(i))):
                unfinished.append((protein, target, start, i))

    print("Missing:", len(unfinished))
    print(unfinished)


def run_align_combine(process, raw_root):
    counter = 0
    for protein, target, start in process:
        if counter == 10:
            break
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, pair)
        conformer_file = os.path.join(pair_path, "{}_lig0-out.maegz".format(target))
        conformers = list(structure.StructureReader(conformer_file))
        if len(conformers) == 1:
            continue
        else:
            counter += 1
        output_path = os.path.join(pair_path, 'conformers')

        combined_file = os.path.join(pair_path, "aligned_to_start_without_hydrogen_conformers.mae")
        with structure.StructureWriter(combined_file) as combined:
            for i in range(len(conformers)):
                aligned_file = os.path.join(output_path, '{}_align_without_hydrogen.mae'.format(i))
                s = list(structure.StructureReader(aligned_file))[0]
                combined.append(s)

        print(len(list(structure.StructureReader(combined_file))))


def search_system_caller(process, raw_root, run_path, docked_prot_file, rotation_search_step_size, no_prot_h):
    counter = 0
    for protein, target, start in process:
        if counter == 10:
            break
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, pair)
        conformer_file = os.path.join(pair_path, "{}_lig0-out.maegz".format(target))
        conformers = list(structure.StructureReader(conformer_file))
        if len(conformers) == 1:
            continue
        else:
            counter += 1
        cmd = 'sbatch -p owners -t 10:00:00 -o {} --wrap="$SCHRODINGER/run python3 systematic_decoy_search.py search ' \
              '{} {} {} --protein {} --target {} --start {} --rotation_search_step_size {}'
        if no_prot_h:
            cmd += ' --remove_prot_h"'
            out_file_name = '{}_{}-to-{}_step_size_{}_no_lig_h_no_prot_h.out'.format(protein, target, start,
                                                                                     rotation_search_step_size)
            os.system(cmd.format(os.path.join(run_path, out_file_name), docked_prot_file, run_path, raw_root, protein,
                                 target, start, rotation_search_step_size))
        else:
            cmd += '"'
            out_file_name = '{}_{}-to-{}_step_size_{}_no_lig_h.out'.format(protein, target, start,
                                                                           rotation_search_step_size)
            os.system(cmd.format(os.path.join(run_path, out_file_name), docked_prot_file, run_path, raw_root, protein,
                                 target, start, rotation_search_step_size))

def get_pocket_res(protein, ligand, dist):
    """
    Given a co-crystallized protein and ligand, extract residues within specified distance of ligand.

    Args:
        protein (Biopython Structure object): receptor protein
        ligand (RDKit Mol object): co-crystallized ligand
        dist (float): distance cutoff for defining binding site

    Returns:
        key_residues (set of Biopython Residue objects): set of key binding site residues
    """
    # get protein coordinates
    prot_coords = []
    for m in list(protein.molecule):
        for r in list(m.residue):
            for a in list(r.atom):
                prot_coords.append(a.xyz)

    # get ligand coordinates
    lig_coords = []
    for m in list(ligand.molecule):
        for r in list(m.residue):
            for a in list(r.atom):
                lig_coords.append(a.xyz)

    kd_tree = scipy.spatial.KDTree(prot_coords)
    key_pts = kd_tree.query_ball_point(lig_coords, r=dist, p=2.0)
    key_pts = set([k for l in key_pts for k in l])

    remove = [i for i in protein.getAtomIndices() if i not in key_pts]
    protein.deleteAtoms(remove)


def run_search(protein, target, start, raw_root, get_clash_rmsd_time, cutoff, rotation_search_step_size, no_prot_h,
               pocket_only, test=False, x_rot=0, y_rot=0, z_rot=0):
    pair = '{}-to-{}'.format(target, start)
    protein_path = os.path.join(raw_root, protein)
    pair_path = os.path.join(protein_path, pair)
    start_lig_file = os.path.join(pair_path, '{}_lig.mae'.format(start))
    start_lig = list(structure.StructureReader(start_lig_file))[0]
    target_lig_file = os.path.join(pair_path, 'ligand_poses', '{}_lig0.mae'.format(target))
    target_lig = list(structure.StructureReader(target_lig_file))[0]
    build.delete_hydrogens(target_lig)
    start_lig_center = list(get_centroid(start_lig))
    prot_file = os.path.join(pair_path, '{}_prot.mae'.format(start))
    prot = list(structure.StructureReader(prot_file))[0]
    print(prot.atom_total)
    if pocket_only:
        get_pocket_res(prot, target_lig, 6)
        print(prot.atom_total)
    if no_prot_h:
        build.delete_hydrogens(prot)
        print(prot.atom_total)

    conformer_file = os.path.join(pair_path, "aligned_to_start_without_hydrogen_conformers.mae")
    conformers = list(structure.StructureReader(conformer_file))
    conformer = create_conformer_decoys(conformers, start_lig_center, target_lig, prot, get_clash_rmsd_time, cutoff,
                                        rotation_search_step_size, protein, target, start, no_prot_h, test, x_rot,
                                        y_rot, z_rot)
    return conformer


def run_test_search(protein, target, start, raw_root, get_clash_rmsd_time, cutoff, rotation_search_step_size,
                    pair_path):
    angles = [i for i in range(-30, 30, rotation_search_step_size)]
    angles = angles[:5]
    x_rot = random.choice(angles)
    y_rot = random.choice(angles)
    z_rot = random.choice(angles)

    conformer = run_search(protein, target, start, raw_root, get_clash_rmsd_time,
                           cutoff, rotation_search_step_size, True, x_rot, y_rot, z_rot)
    start_lig_file = os.path.join(pair_path, '{}_lig.mae'.format(start))
    start_lig = list(structure.StructureReader(start_lig_file))[0]
    start_lig_center = list(get_centroid(start_lig))

    conformer_file = os.path.join(pair_path, "aligned_to_start_conformers.mae")
    base_conf = list(structure.StructureReader(conformer_file))[0]
    grid_loc = [0, 0, 0]
    base_conf_center = list(get_centroid(base_conf))
    transform.translate_structure(base_conf, start_lig_center[0] - base_conf_center[0] + grid_loc[0],
                                  start_lig_center[1] - base_conf_center[1] + grid_loc[1],
                                  start_lig_center[2] - base_conf_center[2] + grid_loc[2])
    base_conf_center = list(get_centroid(base_conf))
    transform.rotate_structure(base_conf, math.radians(x_rot), math.radians(y_rot), math.radians(z_rot),
                               base_conf_center)
    rmsd_val = rmsd.calculate_in_place_rmsd(conformer, conformer.getAtomIndices(), base_conf,
                                            base_conf.getAtomIndices())
    if abs(rmsd_val) < 0.2:
        print("Search works properly", rmsd_val)
    else:
        print("x_rot =", x_rot, "y_rot =", y_rot, "z_rot =", z_rot)
        print("RMSD =", rmsd_val, "but RMSD should equal 0")


def run_combine_search_data(process, raw_root, rotation_search_step_size):
    search_dict = {'protein': [], 'target': [], 'start': [], 'num_conformers': [], 'num_poses_searched': [],
                   'num_correct_poses_found': [], 'time_elapsed': [], 'time_elapsed_per_conformer': []}
    counter = 0
    for protein, target, start in process:
        if counter == 10:
            break
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, pair)
        conformer_file = os.path.join(pair_path, "{}_lig0-out.maegz".format(target))
        conformers = list(structure.StructureReader(conformer_file))
        if len(conformers) == 1:
            continue
        else:
            counter += 1

        search_dict['protein'].append(protein)
        search_dict['target'].append(target)
        search_dict['start'].append(start)
        search_dict['num_conformers'].append(len(conformers))

        out_file_template = '{}_{}-to-{}_step_size_{}_no_lig_h.out'
        out_file = os.path.join(os.getcwd(), 'decoy_timing_data', out_file_template.format(protein, target, start,
                                                                                           rotation_search_step_size))
        f = open(out_file, "r")
        lines = f.readlines()
        data = lines[-1].split(',')
        data = [float(elem.split('=')[1].strip()) for elem in data]
        search_dict['num_poses_searched'].append(data[0])
        search_dict['num_correct_poses_found'].append(data[1])
        search_dict['time_elapsed'].append(data[2])
        search_dict['time_elapsed_per_conformer'].append(data[2] / len(conformers))

    df = pd.DataFrame.from_dict(search_dict)
    df.to_csv(os.path.join(os.getcwd(), 'decoy_timing_data', 'step_size_{}_no_lig_h.csv'.format(
        rotation_search_step_size)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either align or search')
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('--protein', type=str, default='', help='protein name')
    parser.add_argument('--target', type=str, default='', help='target ligand name')
    parser.add_argument('--start', type=str, default='', help='start ligand name')
    parser.add_argument('--n', type=int, default=10, help='number of alignments processed in each job')
    parser.add_argument('--rotation_search_step_size', type=int, default=1, help='step size between each angle '
                                                                                 'checked, in degrees')
    parser.add_argument('--index', type=int, default=-1, help='grid point group index')
    parser.add_argument('--rmsd_cutoff', type=int, default=2, help='rmsd accuracy cutoff between predicted ligand pose '
                                                                   'and true ligand pose')
    parser.add_argument('--num_conformers', type=int, default=300, help='maximum number of conformers considered')
    parser.add_argument('--time_clash_rmsd', dest='get_clash_rmsd_time', action='store_true')
    parser.add_argument('--no_time_clash_rmsd', dest='get_clash_rmsd_time', action='store_false')
    parser.set_defaults(get_clash_rmsd_time=False)
    parser.add_argument('--remove_prot_h', dest='no_prot_h', action='store_true')
    parser.add_argument('--keep_prot_h', dest='no_prot_h', action='store_false')
    parser.set_defaults(no_prot_h=False)
    parser.add_argument('--prot_pocket_only', dest='pocket_only', action='store_true')
    parser.add_argument('--all_prot', dest='pocket_only', action='store_false')
    parser.set_defaults(pocket_only=False)

    args = parser.parse_args()

    random.seed(0)

    if not os.path.exists(args.run_path):
        os.mkdir(args.run_path)

    pair = '{}-to-{}'.format(args.target, args.start)
    protein_path = os.path.join(args.raw_root, args.protein)
    pair_path = os.path.join(protein_path, pair)

    if args.task == 'conformer_all':
        process = get_prots(args.docked_prot_file)
        random.shuffle(process)
        run_conformer_all(process, args.raw_root, args.run_path, args.docked_prot_file)

    elif args.task == 'conformer_group':
        target_lig_file = os.path.join(pair_path, 'ligand_poses', '{}_lig0.mae'.format(args.target))
        gen_ligand_conformers(target_lig_file, pair_path, args.num_conformers)
        if os.path.exists(os.path.join(pair_path, '{}_lig0.log'.format(args.target))):
            os.remove(os.path.join(pair_path, '{}_lig0.log'.format(args.target)))

    if args.task == 'conformer_check':
        process = get_prots(args.docked_prot_file)
        random.shuffle(process)
        run_conformer_check(process, args.raw_root)

    if args.task == 'align_all':
        process = get_prots(args.docked_prot_file)
        random.shuffle(process)
        run_align_all(process, args.raw_root, args.run_path, args.docked_prot_file, args.n)

    elif args.task == 'align_group':
        grouped_files = get_conformer_groups(args.n, args.target, args.start, args.protein, args.raw_root)
        run_align_group(grouped_files, args.index, args.n, args.protein, args.target, args.start, args.raw_root)

    elif args.task == 'align_check':
        process = get_prots(args.docked_prot_file)
        random.shuffle(process)
        run_align_check(process, args.raw_root)

    elif args.task == 'align_combine':
        process = get_prots(args.docked_prot_file)
        random.shuffle(process)
        run_align_combine(process, args.raw_root)

    elif args.task == 'run_search':
        process = get_prots(args.docked_prot_file)
        random.shuffle(process)
        search_system_caller(process, args.raw_root, args.run_path, args.docked_prot_file,
                             args.rotation_search_step_size, args.no_prot_h)

    elif args.task == 'search':
        run_search(args.protein, args.target, args.start, args.raw_root, args.get_clash_rmsd_time, args.rmsd_cutoff,
                   args.rotation_search_step_size, args.no_prot_h, args.pocket_only)

    elif args.task == 'test_search':
        run_test_search(args.protein, args.target, args.start, args.raw_root, args.get_clash_rmsd_time,
                        args.rmsd_cutoff, args.rotation_search_step_size, pair_path)

    elif args.task == 'combine_search_data':
        process = get_prots(args.docked_prot_file)
        random.shuffle(process)
        run_combine_search_data(process, args.raw_root, args.rotation_search_step_size)


if __name__=="__main__":
    main()