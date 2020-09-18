"""
The purpose of this code is to first create the raw directory folder and include the following files
starting protein receptor
starting ligand
target ligand
glide pose viewer file

Then the top glide poses are added

Then the decoys are created

It can be run on sherlock using
$ $SCHRODINGER/run python3 decoy.py all /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /oak/stanford/groups/rondror/projects/ligand-docking/pdbbind_2019/data
$ $SCHRODINGER/run python3 decoy.py group /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /oak/stanford/groups/rondror/projects/ligand-docking/pdbbind_2019/data --index 0
$ $SCHRODINGER/run python3 decoy.py check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /oak/stanford/groups/rondror/projects/ligand-docking/pdbbind_2019/data
$ $SCHRODINGER/run python3 decoy.py delete /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /oak/stanford/groups/rondror/projects/ligand-docking/pdbbind_2019/data
"""

import argparse
import os
import schrodinger.structure as structure
import schrodinger.structutils.transform as transform
from schrodinger.structutils.transform import get_centroid
import numpy as np
import statistics
import pickle
from tqdm import tqdm

class MCSS:
    """
    Reads and writes MCSS features for a ligand pair.

    There are two key phases of the computation:
        (1) Identification of maximum common substructure(s)
        (2) Computation of RMSDs between the substructures in
            docking results.

    Task (1) is accomplished using Schrodinger's canvasMCSS utility.
    Task (2) is accomplished by identifying all matches of the substructure(s)
    from (1) and finding the pair with the mimimum RMSD. This is a subtlely
    difficult task because of symmetry concerns and details of extracting
    substructures.

    MCSS must be at least half the size of the smaller of the ligands
    or no RMSDs are computed.

    A key design decision is to not specify any file names in this class
    (other than those associated with temp files). The implication of this
    is that MCSSController will be completely in control of this task, while
    this class can be dedicated to actually computing the MCSS feature.
    """

    mcss_cmd = ("$SCHRODINGER/utilities/canvasMCS -imae {} -ocsv {}"
                " -stop {} -atomtype C {}")

    def __init__(self, l1, l2):
        """
        l1, l2: string, ligand names
        """
        if l1 > l2: l1, l2 = l2, l1

        self.l1 = l1
        self.l2 = l2
        self.name = "{}-{}".format(l1, l2)

        self.n_l1_atoms = 0
        self.n_l2_atoms = 0
        self.n_mcss_atoms = 0
        self.smarts_l1 = []
        self.smarts_l2 = []
        self.rmsds = {}

        self.tried_small = False

        # Deprecated.
        self.n_mcss_bonds = 0

    def __str__(self):
        return ','.join(map(str,
                            [self.l1, self.l2,
                             self.n_l1_atoms, self.n_l2_atoms, self.n_mcss_atoms, self.n_mcss_bonds,
                             ';'.join(self.smarts_l1), ';'.join(self.smarts_l2), self.tried_small]
                            ))

    def compute_mcss(self, ligands, init_file, mcss_types_file, small=False):
        """
        Compute the MCSS file by calling Schrodinger canvasMCSS.

        Updates instance with MCSSs present in the file
        """
        structure_file = '{}.ligands.mae'.format(init_file)
        mcss_file = '{}.mcss.csv'.format(init_file)
        stwr = StructureWriter(structure_file)
        stwr.append(ligands[self.l1])
        stwr.append(ligands[self.l2])
        stwr.close()
        # set the sizes in atoms of each of the ligands
        self._set_ligand_sizes(structure_file)

        if os.system(self.mcss_cmd.format(structure_file,
                                          mcss_file,
                                          5 if small else 10,
                                          mcss_types_file)):
            assert False, 'MCSS computation failed'
        self._set_mcss(mcss_file)
        self.tried_small = small

        with open(init_file, 'a+') as fp:
            fp.write(str(self) + '\n')

        os.system('rm {} {}'.format(structure_file, mcss_file))

    def _set_ligand_sizes(self, structure_file):
        try:
            refs = [st for st in StructureReader(structure_file)]
        except:
            print('Unable to read MCSS structure file for', self.l1, self.l2)
            return None
        if len(refs) != 2:
            print('Wrong number of structures', self.l1, self.l2)
            return None
        ref1, ref2 = refs
        n_l1_atoms = len([a for a in ref1.atom if a.element != 'H'])
        n_l2_atoms = len([a for a in ref2.atom if a.element != 'H'])

        self.n_l1_atoms = n_l1_atoms
        self.n_l2_atoms = n_l2_atoms

    def _set_mcss(self, mcss_file):
        """
        Updates MCS from the direct output of canvasMCSS.

        Note that there can be multiple maximum common substructures
        of the same size.
        """
        ligs = {}
        n_mcss_atoms = None
        with open(mcss_file) as fp:
            fp.readline()  # Header
            for line in fp:
                smiles, lig, _, _, _, _n_mcss_atoms, _n_mcss_bonds = line.strip().split(',')[:7]
                smarts = line.strip().split(',')[-1]  # There are commas in some of the fields
                _n_mcss_atoms = int(_n_mcss_atoms)

                assert n_mcss_atoms is None or n_mcss_atoms == _n_mcss_atoms, self.name

                if lig not in ligs: ligs[lig] = []
                ligs[lig] += [smarts]
                n_mcss_atoms = _n_mcss_atoms

        if len(ligs) != 2:
            print('Wrong number of ligands in MCSS file', ligs)
            return None
        assert all(smarts for smarts in ligs.values()), ligs

        # MCSS size can change when tautomers change. One particularly prevalent
        # case is when oxyanions are neutralized. Oxyanions are sometimes specified
        # by the smiles string, but nevertheless glide neutralizes them.
        # Can consider initially considering oxyanions and ketones interchangable
        # (see mcss15.typ).
        if self.n_mcss_atoms:
            assert self.n_mcss_atoms <= n_mcss_atoms + 1, 'MCSS size decreased by more than 1.'
            if self.n_mcss_atoms < n_mcss_atoms:
                print(self.name, 'MCSS size increased.')
            if self.n_mcss_atoms > n_mcss_atoms:
                print(self.name, 'MCSS size dencreased by one.')

        self.n_mcss_atoms = n_mcss_atoms

def get_prots(docked_prot_file):
    """
    gets list of all protein, target ligands, and starting ligands in the index file
    :param docked_prot_file: (string) file listing proteins to process
    :return: process (list) list of all protein, target ligands, and starting ligands to process
    """
    process = []
    with open(docked_prot_file) as fp:
        for line in fp:
            if line[0] == '#': continue
            protein, target, start = line.strip().split()
            process.append((protein, target, start))

    return process

def group_files(n, process):
    """
    groups pairs into sublists of size n
    :param n: (int) sublist size
    :param process: (list) list of pairs to process
    :return: grouped_files (list) list of sublists of pairs
    """
    grouped_files = []

    for i in range(0, len(process), n):
        grouped_files += [process[i: i + n]]

    return grouped_files

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

def cartesian_vector(i):
    """
    Generates a random 3D unit vector (direction) with a uniform spherical distribution
    Algo from http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
    :return:
    """
    if i == 0:
        return 1, 0, 0
    elif i == 1:
        return -1, 0, 0
    elif i == 2:
        return 0, 1, 0
    elif i == 3:
        return 0, -1, 0
    elif i == 4:
        return 0, 0, 1
    elif i == 5:
        return 0, 0, -1
    else:
        print('Bad input')


def modify_file(path, name):
    reading_file = open(path, "r")
    file_name = path.split('/')[-1]

    new_file_content = ""
    for line in reading_file:
        if line.strip() == name:
            new_line = line.replace(name, file_name)
        else:
            new_line = line
        new_file_content += new_line
    reading_file.close()

    writing_file = open(path, "w")
    writing_file.write(new_file_content)
    writing_file.close()

def create_mcss_file(path, ligand, save_folder):
    reading_file = open(path, "r")

    new_file_content = ""
    for line in reading_file:
        new_line = line.replace("_pro_ligand", ligand)
        new_file_content += new_line + "\n"
    reading_file.close()

    writing_path = os.path.join(save_folder, '{}.mae'.format(ligand))
    writing_file = open(writing_path, "w")
    writing_file.write(new_file_content)
    writing_file.close()
    return writing_path

def compute_protein_mcss(ligands, pair_path):
    init_file = '{}/{}-to-{}_mcss.csv'.format(pair_path, ligands[0], ligands[1])
    for i in range(len(ligands)):
        for j in range(i + 1, len(ligands)):
            l1, l2 = ligands[i], ligands[j]
            l1_path = '{}/ligand_poses/{}_lig0.mae'.format(pair_path, l1)
            new_l1_path = create_mcss_file(l1_path, l1, pair_path)
            l2_path = '{}/{}_lig.mae'.format(pair_path, l2)
            new_l2_path = create_mcss_file(l2_path, l2, pair_path)
            mcss_types_file = 'mcss_type_file.typ'
            mcss = MCSS(l1, l2)
            with structure.StructureReader(new_l1_path) as ligand1, structure.StructureReader(new_l2_path) as ligand2:
                ligands = {l1: next(ligand1), l2: next(ligand2)}
                mcss.compute_mcss(ligands, init_file, mcss_types_file)
            os.system('rm {} {}'.format(new_l1_path, new_l2_path))

def create_decoys(lig_file, max_decoys, mean_translation, stdev_translation, min_angle, max_angle):
    """
    creates MAX_DECOYS number of translated/rotated decoys
    :param lig_file: (string) file of glide ligand pose that will be translated/rotated
    :return:
    """
    code = lig_file.split('/')[-1].split('_')[-1]
    if code == 'lig0.mae':
        modify_file(lig_file, '_pro_ligand')
    else:
        modify_file(lig_file, '_ligand')
    for i in range(max_decoys):
        s = list(structure.StructureReader(lig_file))[0]

        # translation
        x, y, z = random_three_vector()
        dist = np.random.normal(mean_translation, stdev_translation)
        transform.translate_structure(s, x * dist, y * dist, z * dist)

        # rotation
        x_angle = np.random.uniform(min_angle, max_angle)
        y_angle = np.random.uniform(min_angle, max_angle)
        z_angle = np.random.uniform(min_angle, max_angle)
        rot_center = list(get_centroid(s))
        transform.rotate_structure(s, x_angle, y_angle, z_angle, rot_center)

        decoy_file = lig_file[:-4] + chr(ord('a')+i) + '.mae'
        with structure.StructureWriter(decoy_file) as decoy:
            decoy.append(s)
        if code == 'lig0.mae':
            modify_file(decoy_file, lig_file.split('/')[-1])
        else:
            modify_file(decoy_file, lig_file.split('/')[-1])

def create_cartesian_decoys(lig_file):
    """
    creates MAX_DECOYS number of translated/rotated decoys
    :param lig_file: (string) file of glide ligand pose that will be translated/rotated
    :return:
    """
    code = lig_file.split('/')[-1].split('_')[-1]
    if code == 'lig0.mae':
        modify_file(lig_file, '_pro_ligand')
    else:
        modify_file(lig_file, '_ligand')
    for i in range(6):
        s = list(structure.StructureReader(lig_file))[0]

        # translation
        x, y, z = cartesian_vector(i)
        transform.translate_structure(s, x, y, z)

        decoy_file = lig_file[:-4] + chr(ord('a')+i) + '.mae'
        with structure.StructureWriter(decoy_file) as decoy:
            decoy.append(s)
        if code == 'lig0.mae':
            modify_file(decoy_file, lig_file.split('/')[-1])
        else:
            modify_file(decoy_file, lig_file.split('/')[-1])

def run_all(docked_prot_file, run_path, raw_root, data_root, grouped_files, n):
    """
    submits sbatch script to create decoys for each protein, target, start group
    :param docked_prot_file: (string) file listing proteins to process
    :param run_path: (string) directory where script and output files will be written
    :param raw_root: (string) directory where raw data will be placed
    :param data_root: (string) pdbbind directory where raw data will be obtained
    :param grouped_files: (list) list of protein, target, start groups
    :return:
    """
    for i, group in enumerate(grouped_files):
        cmd = 'sbatch -p owners -t 1:00:00 -o {} --wrap="$SCHRODINGER/run python3 decoy.py group {} {} {} {} --n {} ' \
              '--index {}"'
        os.system(cmd.format(os.path.join(run_path, 'decoy{}.out'.format(i)), docked_prot_file,
                             run_path, raw_root, data_root, n, i))

def run_group(grouped_files, raw_root, data_root, index, max_poses, decoy_type, max_decoys, mean_translation,
              stdev_translation, min_angle, max_angle):
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
    for protein, target, start in grouped_files[index]:
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, pair)
        pose_path = os.path.join(pair_path, 'ligand_poses')
        dock_root = os.path.join(data_root, '{}/docking/sp_es4/{}'.format(protein, pair))
        struct_root = os.path.join(data_root, '{}/structures/aligned'.format(protein))

        # # create folders
        # if not os.path.exists(raw_root):
        #     os.mkdir(raw_root)
        # if not os.path.exists(protein_path):
        #     os.mkdir(protein_path)
        # if not os.path.exists(pair_path):
        #     os.mkdir(pair_path)
        # if not os.path.exists(pose_path):
        #     os.mkdir(pose_path)
        #
        # # add basic files
        # if not os.path.exists('{}/{}_prot.mae'.format(pair_path, start)):
        #     os.system('cp {}/{}_prot.mae {}/{}_prot.mae'.format(struct_root, start, pair_path, start))
        # if not os.path.exists('{}/{}_lig.mae'.format(pair_path, start)):
        #     os.system('cp {}/{}_lig.mae {}/{}_lig.mae'.format(struct_root, start, pair_path, start))
        # if not os.path.exists('{}/{}_lig0.mae'.format(pair_path, target)):
        #     os.system('cp {}/{}_lig.mae {}/{}_lig0.mae'.format(struct_root, target, pose_path, target))

        # add combine glide poses
        pv_file = '{}/{}_glide_pv.maegz'.format(pair_path, pair)
        if not os.path.exists(pv_file):
            os.system('cp {}/{}_pv.maegz {}'.format(dock_root, pair, pv_file))

        # # extract glide poses and create decoys
        # num_poses = len(list(structure.StructureReader(pv_file)))
        # for i in range(num_poses):
        #     if i == max_poses:
        #         break
        #     lig_file = os.path.join(pose_path, '{}_lig{}.mae'.format(target, i))
        #     if i != 0:
        #         with structure.StructureWriter(lig_file) as all_file:
        #             all_file.append(list(structure.StructureReader(pv_file))[i])
        #     if decoy_type == 'cartesian':
        #         create_cartesian_decoys(lig_file)
        #     elif decoy_type == 'random':
        #         create_decoys(lig_file, max_decoys, mean_translation, stdev_translation, min_angle, max_angle)
        #
        # # combine ligands
        # with structure.StructureWriter('{}/{}_merge_pv.mae'.format(pair_path, pair)) as all_file:
        #     for file in os.listdir(pose_path):
        #         if file[-3:] == 'mae':
        #             pv = list(structure.StructureReader(os.path.join(pose_path, file)))
        #             all_file.append(pv[0])
        #
        # # compute mcss
        # compute_protein_mcss([target, start], pair_path)
        #
        # # zip file
        # os.system('gzip -f {}/{}_merge_pv.mae'.format(pair_path, pair, pair_path, pair))

def run_check(docked_prot_file, raw_root, max_poses, max_decoys):
    """
    check if all files are created
    :param docked_prot_file: (string) file listing proteins to process
    :param raw_root: (string) directory where raw data will be placed
    :param max_poses: (int) maximum number of glide poses considered
    :param max_decoys: (int) maximum number of decoys created per glide pose
    :return:
    """
    process = []
    num_pairs = 0
    with open(docked_prot_file) as fp:
        for line in tqdm(fp, desc='protein, target, start groups'):
            if line[0] == '#': continue
            protein, target, start = line.strip().split()
            pair = '{}-to-{}'.format(target, start)
            num_pairs += 1
            protein_path = os.path.join(raw_root, protein)
            pair_path = os.path.join(protein_path, pair)
            pose_path = os.path.join(pair_path, 'ligand_poses')
            pv_file = os.path.join(pair_path, '{}_glide_pv.maegz'.format(pair))

            # if not os.path.exists('{}/{}_prot.mae'.format(pair_path, start)):
            #     process.append((protein, target, start))
            #     print('{}/{}_prot.mae'.format(pair_path, start))
            #     continue
            # if not os.path.exists('{}/{}_lig.mae'.format(pair_path, start)):
            #     process.append((protein, target, start))
            #     print('{}/{}_lig.mae'.format(pair_path, start))
            #     continue
            # if not os.path.exists('{}/{}_lig0.mae'.format(pose_path, target)):
            #     process.append((protein, target, start))
            #     print('{}/{}_lig0.mae'.format(pose_path, target))
            #     continue
            if not os.path.exists(pv_file):
                process.append((protein, target, start))
                # print(pv_file)
                # continue

            # num_poses = min(max_poses, len(list(structure.StructureReader(pv_file))))
            # if not os.path.exists(os.path.join(pose_path, '{}_lig{}.mae'.format(target, num_poses - 1))):
            #     process.append((protein, target, start))
            #     print(os.path.join(pose_path, '{}_lig{}.mae'.format(target, num_poses - 1)))
            #     continue
            # for i in range(max_decoys):
            #     if not os.path.exists(os.path.join(pose_path, '{}_lig{}.mae'.format(target, str(num_poses - 1) +
            #                                                                                 chr(ord('a') + i)))):
            #         process.append((protein, target, start))
            #         print(os.path.join(pose_path, '{}_lig{}.mae'.format(target, str(num_poses - 1) + chr(ord('a') + i))))
            #         break
            #
            # if not os.path.exists(os.path.join(pair_path, '{}_mcss.csv'.format(pair))):
            #     process.append((protein, target, start))
            #     continue
            # if not os.path.exists(os.path.join(pair_path, '{}/{}_merge_pv.mae.gz'.format(pair_path, pair))):
            #     process.append((protein, target, start))
            #     continue


    print('Missing', len(process), '/', num_pairs)
    # print(process)

def run_all_dist_check(docked_prot_file, run_path, raw_root, data_root, grouped_files):
    """
    submits sbatch script to check mean distance of displacement for decoys for each protein, target, start group
    :param docked_prot_file: (string) file listing proteins to process
    :param run_path: (string) directory where script and output files will be written
    :param raw_root: (string) directory where raw data will be placed
    :param data_root: (string) pdbbind directory where raw data will be obtained
    :param grouped_files: (list) list of protein, target, start groups
    :return:
    """
    for i, group in enumerate(grouped_files):
        cmd = 'sbatch -p owners -t 1:00:00 -o {} --wrap="$SCHRODINGER/run python3 decoy.py group_dist_check {} {} {} {} ' \
              '--index {}"'
        os.system(cmd.format(os.path.join(run_path, 'dist{}.out'.format(i)), docked_prot_file,
                             run_path, raw_root, data_root, i))

def run_group_dist_check(grouped_files, raw_root, index, dist_dir, max_poses, max_decoys):
    """
    checks mean distance of displacement for decoys for each protein, target, start group
    :param grouped_files: (list) list of protein, target, start groups
    :param raw_root: (string) directory where raw data will be placed
    :param index: (int) group number
    :param dist_dir: (string) directiory to place distances
    :param max_poses: (int) maximum number of glide poses considered
    :param max_decoys: (int) maximum number of decoys created per glide pose
    :return:
    """
    save = []
    for protein, target, start in grouped_files[index]:
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
        pose_path = os.path.join(pair_path, 'ligand_poses')
        pv_file = os.path.join(pair_path, '{}-to-{}_pv.maegz'.format(target, start))
        num_poses = len(list(structure.StructureReader(pv_file)))
        means = []

        for i in range(num_poses):
            if i == max_poses:
                break
            lig_file = os.path.join(pose_path, '{}_lig{}.mae'.format(target, i))
            s = list(structure.StructureReader(lig_file))[0]
            c = get_centroid(s)
            dists = []

            for j in range(max_decoys):
                decoy_file = lig_file[:-4] + chr(ord('a') + j) + '.mae'
                decoy = list(structure.StructureReader(decoy_file))[0]
                dists.append(transform.get_vector_magnitude(c - get_centroid(decoy)))

            means.append(statistics.mean(dists))

        save.append(statistics.mean(means))

    outfile = open(os.path.join(dist_dir, '{}.pkl'.format(index)), 'wb')
    pickle.dump(save, outfile)
    print(save)

def run_check_dist_check(grouped_files, dist_dir):
    """
    check if all dist files created and if all means are appropriate
    :param grouped_files: (list) list of protein, target, start groups
    :param dist_dir: (string) directiory to place distances
    :return:
    """
    if len(os.listdir(dist_dir)) != len(grouped_files):
        print('Not all files created')
    else:
        print('All files created')

    errors = []
    for i in range(len(grouped_files)):
        infile = open(os.path.join(dist_dir, '{}.pkl'.format(i)), 'rb')
        vals = pickle.load(infile)
        infile.close()

        for j in vals:
            if j > 2 or j < -1:
                print(vals)
                errors.append(i)
                break

    print('Potential errors', len(errors), '/', len(grouped_files))
    print(errors)


def run_all_name_check(docked_prot_file, run_path, raw_root, data_root, grouped_files):
    """
    submits sbatch script to check names of decoys for each protein, target, start group
    :param docked_prot_file: (string) file listing proteins to process
    :param run_path: (string) directory where script and output files will be written
    :param raw_root: (string) directory where raw data will be placed
    :param data_root: (string) pdbbind directory where raw data will be obtained
    :param grouped_files: (list) list of protein, target, start groups
    :return:
    """
    for i, group in enumerate(grouped_files):
        cmd = 'sbatch -p owners -t 1:00:00 -o {} --wrap="$SCHRODINGER/run python3 decoy.py group_name_check {} {} {} {} ' \
              '--index {}"'
        os.system(cmd.format(os.path.join(run_path, 'name{}.out'.format(i)), docked_prot_file,
                             run_path, raw_root, data_root, i))


def run_group_name_check(grouped_files, raw_root, index, name_dir, max_poses, max_decoys):
    """
    checks names of decoys for each protein, target, start group
    :param grouped_files: (list) list of protein, target, start groups
    :param raw_root: (string) directory where raw data will be placed
    :param index: (int) group number
    :param name_dir: (string) directiory to place unfinished protein, target, start groups
    :param max_poses: (int) maximum number of glide poses considered
    :param max_decoys: (int) maximum number of decoys created per glide pose
    :return:
    """
    unfinished = []
    for protein, target, start in grouped_files[index]:
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
        pose_path = os.path.join(pair_path, 'ligand_poses')
        pv_file = os.path.join(pair_path, '{}-to-{}_glide_pv.maegz'.format(target, start))
        num_poses = len(list(structure.StructureReader(pv_file)))

        for i in range(num_poses):
            if i == max_poses:
                break
            lig_file = os.path.join(pose_path, '{}_lig{}.mae'.format(target, i))
            found = False
            with open(lig_file, "r") as f:
                file_name = lig_file.split('/')[-1]
                for line in f:
                    if line.strip() == file_name:
                        found = True
            if not found:
                print(lig_file)
                unfinished.append((protein, target, start))
                break
            else:
                for j in range(max_decoys):
                    decoy_file = lig_file[:-4] + chr(ord('a') + j) + '.mae'
                    found = False
                    with open(decoy_file, "r") as f:
                        file_name = decoy_file.split('/')[-1]
                        for line in f:
                            if line.strip() == file_name:
                                found = True
                    if not found:
                        print(decoy_file)
                        unfinished.append((protein, target, start))
                        break
            if not found:
                break
        break

    outfile = open(os.path.join(name_dir, '{}.pkl'.format(index)), 'wb')
    pickle.dump(unfinished, outfile)
    print(unfinished)


def run_check_name_check(process, grouped_files, name_dir):
    """
    check if all dist files created and if all means are appropriate
    :param process: (list) list of all protein, target, start
    :param grouped_files: (list) list of protein, target, start groups
    :param name_dir: (string) directiory to place unfinished protein, target, start groups
    :return:
    """
    if len(os.listdir(name_dir)) != len(grouped_files):
        print('Not all files created')
    else:
        print('All files created')

    errors = []
    for i in range(len(grouped_files)):
        infile = open(os.path.join(name_dir, '{}.pkl'.format(i)), 'rb')
        unfinished = pickle.load(infile)
        infile.close()
        errors.extend(unfinished)

    print('Errors', len(errors), '/', len(process))
    print(errors)

def run_delete(grouped_files, run_path, raw_root):
    """
    delete all folders in raw_root
    :param grouped_files: (list) list of protein, target, start groups
    :param run_path: (string) directory where script and output files will be written
    :param raw_root: (string) directory where raw data will be placed
    :return:
    """
    for i, group in enumerate(grouped_files):
        with open(os.path.join(run_path, 'delete{}_in.sh'.format(i)), 'w') as f:
            f.write('#!/bin/bash\n')
            for protein in group:
                f.write('rm -r {}\n'.format(os.path.join(raw_root, protein)))

        os.chdir(run_path)
        os.system('sbatch -p owners -t 02:00:00 -o delete{}.out delete{}_in.sh'.format(i, i))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either all, group, check, '
                                               'all_dist_check, group_dist_check, check_dist_check, '
                                               'all_name_check, group_name_check, check_name_check, or delete')
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('data_root', type=str, help='pdbbind directory where raw data will be obtained')
    parser.add_argument('--index', type=int, default=-1, help='for group task, group number')
    parser.add_argument('--dist_dir', type=str, default=os.path.join(os.getcwd(), 'dists'),
                        help='for all_dist_check and group_dist_check task, directiory to place distances')
    parser.add_argument('--name_dir', type=str, default=os.path.join(os.getcwd(), 'names'),
                        help='for all_name_check and group_name_check task, directiory to place unfinished protein, '
                             'target, start groups')
    parser.add_argument('--n', type=int, default=3, help='number of protein, target, start groups processed in '
                                                          'group task')
    parser.add_argument('--max_poses', type=int, default=100, help='maximum number of glide poses considered')
    parser.add_argument('--decoy_type', type=str, default='random', help='either cartesian or random')
    parser.add_argument('--max_decoys', type=int, default=10, help='maximum number of decoys created per glide pose')
    parser.add_argument('--mean_translation', type=int, default=0, help='mean distance decoys are translated')
    parser.add_argument('--stdev_translation', type=int, default=1, help='stdev of distance decoys are translated')
    parser.add_argument('--min_angle', type=float, default=- np.pi / 6, help='minimum angle decoys are rotated')
    parser.add_argument('--max_angle', type=float, default=np.pi / 6, help='maximum angle decoys are rotated')
    args = parser.parse_args()

    if not os.path.exists(args.run_path):
        os.mkdir(args.run_path)

    if args.task == 'all':
        process = get_prots(args.docked_prot_file)
        grouped_files = group_files(args.n, process)
        run_all(args.docked_prot_file, args.run_path, args.raw_root, args.data_root, grouped_files, args.n)

    if args.task == 'group':
        process = get_prots(args.docked_prot_file)
        grouped_files = group_files(args.n, process)
        run_group(grouped_files, args.raw_root, args.data_root, args.index, args.max_poses, args.decoy_type,
                  args.max_decoys, args.mean_translation, args.stdev_translation, args.min_angle, args.max_angle)

    if args.task == 'check':
        run_check(args.docked_prot_file, args.raw_root, args.max_poses, args.max_decoys)

    if args.task == 'all_dist_check':
        if not os.path.exists(args.dist_dir):
            os.mkdir(args.dist_dir)

        process = get_prots(args.docked_prot_file)
        grouped_files = group_files(args.n, process)
        run_all_dist_check(args.docked_prot_file, args.run_path, args.raw_root, args.data_root, grouped_files)

    if args.task == 'group_dist_check':
        if not os.path.exists(args.dist_dir):
            os.mkdir(args.dist_dir)

        process = get_prots(args.docked_prot_file)
        grouped_files = group_files(args.n, process)
        run_group_dist_check(grouped_files, args.raw_root, args.index, args.dist_dir, args.max_poses, args.max_decoys)

    if args.task == 'check_dist_check':
        process = get_prots(args.docked_prot_file)
        grouped_files = group_files(args.n, process)
        run_check_dist_check(grouped_files, args.dist_dir)

    if args.task == 'all_name_check':
        if not os.path.exists(args.name_dir):
            os.mkdir(args.name_dir)

        process = get_prots(args.docked_prot_file)
        grouped_files = group_files(args.N, process)
        run_all_name_check(args.docked_prot_file, args.run_path, args.raw_root, args.data_root, grouped_files)

    if args.task == 'group_name_check':
        process = get_prots(args.docked_prot_file)
        grouped_files = group_files(args.n, process)
        run_group_name_check(grouped_files, args.raw_root, args.index, args.name_dir, args.max_poses, args.max_decoys)

    if args.task == 'check_name_check':
        process = get_prots(args.docked_prot_file)
        grouped_files = group_files(args.n, process)
        run_check_name_check(process, grouped_files, args.dist_dir)

    if args.task == 'delete':
        process = os.listdir(args.raw_root)
        grouped_files = group_files(args.n, process)
        run_delete(grouped_files, args.run_path, args.raw_root)

if __name__=="__main__":
    main()