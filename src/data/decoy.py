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
        # process = get_prots(args.docked_prot_file)
        process = [('P00883', '1ado', '2ot1'), ('P00883', '2ot1', '1ado'), ('P00915', '1azm', '3lxe'), ('P00915', '2nmx', '6f3b'), ('P00915', '2nn1', '3lxe'), ('P00915', '2nn7', '6evr'), ('P00915', '3lxe', '6faf'), ('P00915', '6evr', '6g3v'), ('P00915', '6ex1', '6evr'), ('P00915', '6f3b', '2nmx'), ('P00915', '6faf', '2nn7'), ('P00915', '6g3v', '3lxe'), ('P00918', '1avn', '1bnu'), ('P00918', '1bcd', '1bnt'), ('P00918', '1bn1', '1bnn'), ('P00918', '1bn3', '1bnq'), ('P00918', '1bn4', '1bnt'), ('P00918', '1bnn', '1bnv'), ('P00918', '1bnq', '1bnu'), ('P00918', '1bnt', '1bn1'), ('P00918', '1bnu', '1bnn'), ('P00918', '1bnv', '1bnq'), ('P00929', '1tjp', '2clh'), ('P00929', '2cle', '2clh'), ('P00929', '2clh', '2cli'), ('P00929', '2cli', '2clh'), ('P00929', '2clk', '1tjp'), ('P01011', '5om2', '6ftp'), ('P01011', '5om3', '5om2'), ('P01011', '5om7', '5om2'), ('P01011', '6ftp', '5om2'), ('P01112', '4ury', '6d5e'), ('P01112', '4urz', '4ury'), ('P01112', '6d55', '6d5g'), ('P01112', '6d56', '4ury'), ('P01112', '6d5e', '6d5g'), ('P01112', '6d5g', '6d55'), ('P01112', '6d5h', '6d56'), ('P01112', '6d5j', '6d5h'), ('P01116', '4dst', '4epy'), ('P01116', '4dsu', '4dst'), ('P01116', '4epy', '4dst'), ('P01116', '6fa4', '4epy'), ('P01724', '1dl7', '1oar'), ('P01724', '1oar', '1dl7'), ('P01834', '1a4k', '1i7z'), ('P01857', '1aj7', '1gaf'), ('P01857', '1gaf', '1aj7'), ('P01892', '3qfd', '5jzi'), ('P01892', '5isz', '5jzi'), ('P01892', '5jzi', '3qfd'), ('P01901', '1fo0', '1g7q'), ('P01901', '1fzj', '3p9m'), ('P01901', '1fzk', '1fzo'), ('P01901', '1fzm', '1fzj'), ('P01901', '1g7q', '3p9l'), ('P01901', '3p9l', '1fo0'), ('P01901', '3p9m', '1fzo'), ('P01901', '4pg9', '1fzk'), ('P02701', '2a5b', '2jgs'), ('P02701', '2a5c', '2a5b'), ('P02701', '2a8g', '2a5c'), ('P02701', '2jgs', '2a5b'), ('P02743', '2w08', '3kqr'), ('P02743', '3kqr', '4ayu'), ('P02743', '4avs', '3kqr'), ('P02743', '4ayu', '3kqr'), ('P02754', '1gx8', '3uew'), ('P02754', '3nq3', '3uew'), ('P02754', '3nq9', '3uew'), ('P02754', '3ueu', '3uex'), ('P02754', '3uev', '3nq3'), ('P02754', '3uew', '3ueu'), ('P02754', '3uex', '3nq9'), ('P02754', '4gny', '2gj5'), ('P02754', '6ge7', '3ueu'), ('P02766', '1bm7', '2f7i'), ('P02766', '1e4h', '2b9a'), ('P02766', '2b9a', '2f7i'), ('P02766', '2f7i', '1bm7'), ('P02766', '2g5u', '2f7i'), ('P02766', '3cfn', '1bm7'), ('P02766', '3cft', '1bm7'), ('P02766', '3kgt', '3kgu'), ('P02766', '3kgu', '1e4h'), ('P02766', '3nee', '1e4h'), ('P02768', '1hk4', '6ezq'), ('P02768', '6ezq', '1hk4'), ('P02791', '3f33', '3u90'), ('P02829', '1amw', '2weq'), ('P02829', '1bgq', '2weq'), ('P02829', '2cgf', '2vwc'), ('P02829', '2fxs', '2weq'), ('P02829', '2iwx', '2cgf'), ('P02829', '2vw5', '1amw'), ('P02829', '2vwc', '2iwx'), ('P02829', '2weq', '2yge'), ('P02829', '2wer', '2yge'), ('P02829', '2yge', '1bgq'), ('P02911', '1laf', '1lag'), ('P02911', '1lag', '1lah'), ('P02911', '1lah', '1lst'), ('P02911', '1lst', '1lag'), ('P02925', '1drj', '1drk'), ('P02925', '1drk', '1drj'), ('P02925', '2dri', '1drk'), ('P03176', '1e2k', '1e2l'), ('P03176', '1e2l', '1e2k'), ('P03366', '1ajv', '1ec1'), ('P03366', '1ajx', '1d4i'), ('P03366', '1d4h', '1ec1'), ('P03366', '1d4i', '1d4j'), ('P03366', '1d4j', '1d4h'), ('P03366', '1ebw', '1ec0'), ('P03366', '1eby', '1ec1'), ('P03366', '1ebz', '1eby'), ('P03366', '1ec0', '1d4h'), ('P03366', '1ec1', '1ebw'), ('P03367', '1a94', '1dif'), ('P03367', '1d4y', '1iiq'), ('P03367', '1hpo', '1mrw'), ('P03367', '1hpx', '1a94'), ('P03367', '1iiq', '1lzq'), ('P03367', '1lzq', '1a94'), ('P03367', '1m0b', '1hpo'), ('P03367', '1mrw', '1dif'), ('P03368', '1a9m', '1gnm'), ('P03368', '1gnm', '1g35'), ('P03368', '1gnn', '1zp8'), ('P03368', '1gno', '1zp8'), ('P03368', '1zp8', '1gnm'), ('P03368', '1zpa', '2hs1'), ('P03368', '2hs1', '1zpa'), ('P03368', '2i4d', '1gnm'), ('P03369', '1aid', '1z1h'), ('P03369', '1b6j', '1z1h'), ('P03369', '1b6k', '1d4l'), ('P03369', '1b6l', '1aid'), ('P03369', '1d4k', '1kzk'), ('P03369', '1d4l', '1d4k'), ('P03369', '1kzk', '3aid'), ('P03369', '1mtr', '1kzk'), ('P03369', '1z1h', '1d4k'), ('P03369', '3aid', '1aid'), ('P03372', '1qkt', '2qe4'), ('P03372', '2p15', '2qe4'), ('P03372', '2pog', '2qe4'), ('P03372', '4mgd', '2pog'), ('P03472', '1f8b', '2qwd'), ('P03472', '1f8c', '2qwe'), ('P03472', '1f8d', '2qwf'), ('P03472', '1f8e', '1f8d'), ('P03472', '2qwb', '1f8c'), ('P03472', '2qwc', '2qwf'), ('P03472', '2qwd', '2qwf'), ('P03472', '2qwe', '2qwd'), ('P03472', '2qwf', '1f8e'), ('P03951', '4cr5', '4cr9'), ('P03951', '4cr9', '4x6m'), ('P03951', '4cra', '4ty6'), ('P03951', '4crb', '4ty7'), ('P03951', '4crc', '4ty6'), ('P03951', '4crf', '4cr9'), ('P03951', '4ty6', '4cr9'), ('P03951', '4ty7', '4cra'), ('P03951', '4x6m', '4crb'), ('P03951', '4x6n', '4crc'), ('P03956', '1cgl', '966c'), ('P03956', '966c', '1cgl'), ('P03958', '1add', '1fkw'), ('P03958', '1fkw', '1add'), ('P04035', '3cct', '3cd7'), ('P04035', '3ccw', '3cda'), ('P04035', '3ccz', '3cd7'), ('P04035', '3cd0', '3cct'), ('P04035', '3cd5', '3cda'), ('P04035', '3cd7', '3cd0'), ('P04035', '3cda', '3cct'), ('P04035', '3cdb', '3cda'), ('P04058', '1e66', '1h23'), ('P04058', '1gpk', '1h23'), ('P04058', '1gpn', '5nau'), ('P04058', '1h22', '1gpk'), ('P04058', '1h23', '1gpk'), ('P04058', '3zv7', '1gpk'), ('P04058', '5bwc', '5nap'), ('P04058', '5nap', '1gpn'), ('P04058', '5nau', '3zv7'), ('P04117', '1adl', '2qm9'), ('P04117', '1g74', '2ans'), ('P04117', '2ans', '2qm9'), ('P04117', '2qm9', '1adl'), ('P04117', '3hk1', '1adl'), ('P04150', '4p6w', '4p6x'), ('P04150', '4p6x', '4p6w'), ('P04278', '1kdk', '1lhu'), ('P04278', '1lhu', '1kdk'), ('P04584', '1hii', '6upj'), ('P04584', '1hsh', '1ivp'), ('P04584', '1ivp', '1hii'), ('P04584', '5upj', '1ivp'), ('P04584', '6upj', '1hii'), ('P04585', '1a30', '1hwr'), ('P04585', '1bv9', '1bwb'), ('P04585', '1bwa', '1bwb'), ('P04585', '1dmp', '1bwb'), ('P04585', '1hvh', '1bwb'), ('P04585', '1hwr', '1dmp'), ('P04585', '1hxb', '1bwa'), ('P04587', '1bdq', '2aog'), ('P04587', '1g2k', '1hpv'), ('P04587', '1hpv', '2aoe'), ('P04587', '1tcx', '1hvs'), ('P04587', '2aoc', '1tcx'), ('P04587', '2aod', '1hpv'), ('P04587', '2aoe', '1hvj'), ('P04587', '2aog', '1bdq'), ('P04637', '2vuk', '5aba'), ('P04637', '4agl', '4agm'), ('P04637', '4agm', '4agq'), ('P04637', '4agn', '4agl'), ('P04637', '4ago', '4agl'), ('P04637', '4agq', '4agp'), ('P04637', '5aba', '4agl'), ('P04637', '5aoi', '4ago'), ('P04642', '4aj4', '4ajl'), ('P04642', '4aje', '4aj4'), ('P04642', '4aji', '4aje'), ('P04642', '4ajl', '4aji'), ('P04746', '1u33', '1xd0'), ('P04746', '1xd0', '3old'), ('P04746', '3old', '1xd0'), ('P04746', '4gqq', '3old'), ('P04746', '4gqr', '4gqq'), ('P04789', '1iih', '1trd'), ('P04789', '1kv5', '4tim'), ('P04789', '1trd', '1kv5'), ('P04789', '2j27', '1kv5'), ('P04789', '2v2c', '1iih'), ('P04789', '2v2h', '2v2c'), ('P04789', '4tim', '1iih'), ('P04816', '1usi', '1usk'), ('P04816', '1usk', '1usi'), ('P04905', '2gst', '3gst'), ('P04905', '3gst', '2gst'), ('P04995', '3hl8', '3hp9'), ('P04995', '3hp9', '3hl8'), ('P05089', '3f80', '3mfv'), ('P05089', '3kv2', '3lp7'), ('P05089', '3lp4', '3mjl'), ('P05089', '3lp7', '3mfw'), ('P05089', '3mfv', '3lp7'), ('P05089', '3mfw', '3kv2'), ('P05089', '3mjl', '3mfv'), ('P05091', '4kwf', '4kwg'), ('P05091', '4kwg', '4kwf'), ('P05413', '1hmr', '4tkb'), ('P05413', '1hms', '4tjz'), ('P05413', '1hmt', '1hmr'), ('P05413', '3wvm', '1hms'), ('P05413', '4tjz', '3wvm'), ('P05413', '4tkb', '1hmt'), ('P05413', '4tkh', '5hz9'), ('P05413', '4tkj', '4tjz'), ('P05413', '5hz9', '1hms'), ('P05543', '2xn3', '2xn5'), ('P05543', '2xn5', '2xn3'), ('P05981', '1o5e', '1p57'), ('P05981', '1p57', '1o5e'), ('P06202', '1b05', '1b40'), ('P06202', '1b0h', '1b3f'), ('P06202', '1b1h', '1b3h'), ('P06202', '1b2h', '1b3g'), ('P06202', '1b32', '1b3l'), ('P06202', '1b3f', '1b3h'), ('P06202', '1b3g', '1b3l'), ('P06202', '1b3h', '1b2h'), ('P06202', '1b3l', '1b3g'), ('P06202', '1b40', '1b2h'), ('P06239', '1bhf', '1lkk'), ('P06239', '1lkk', '1lkl'), ('P06239', '1lkl', '1lkk'), ('P06276', '6ep4', '6eqp'), ('P06276', '6eqp', '6ep4'), ('P06401', '1a28', '1sr7'), ('P06730', '2v8w', '5ei3'), ('P06730', '4tpw', '2v8w'), ('P06730', '5ei3', '4tpw'), ('P06875', '1ai4', '1ai5'), ('P06875', '1ai5', '1ajq'), ('P06875', '1ai7', '1ai4'), ('P06875', '1ajn', '1ai5'), ('P06875', '1ajp', '1ai4'), ('P06875', '1ajq', '1ai7'), ('P07267', '1fq5', '2jxr'), ('P07267', '2jxr', '1fq5'), ('P07445', '1e3v', '1ogx'), ('P07445', '1ogx', '5g2g'), ('P07445', '5g2g', '1ogx'), ('P07711', '3h89', '3h8b'), ('P07711', '3h8b', '3h89'), ('P07900', '1yc1', '2xab'), ('P07900', '1yc4', '2xdk'), ('P07900', '1yet', '2uwd'), ('P07900', '2qg0', '1yc1'), ('P07900', '2qg2', '1yet'), ('P07900', '2uwd', '1yc4'), ('P07900', '2xab', '1yc1'), ('P07900', '2xdk', '2xdx'), ('P07900', '2xdl', '1yc4'), ('P07900', '2xdx', '2xdk'), ('P07986', '1fh7', '1fh8'), ('P07986', '1fh8', '1fhd'), ('P07986', '1fh9', '1j01'), ('P07986', '1fhd', '1fh9'), ('P07986', '1j01', '1fhd'), ('P08191', '1uwf', '4x5q'), ('P08191', '4css', '4x50'), ('P08191', '4cst', '4x5r'), ('P08191', '4lov', '4cst'), ('P08191', '4x50', '4x5p'), ('P08191', '4x5p', '4css'), ('P08191', '4x5q', '4css'), ('P08191', '4x5r', '4css'), ('P08191', '4xo8', '4lov'), ('P08235', '2oax', '5mwy'), ('P08235', '5l7e', '2oax'), ('P08235', '5l7g', '5l7h'), ('P08235', '5mwp', '5l7g'), ('P08235', '5mwy', '5mwp'), ('P08238', '5uc4', '5ucj'), ('P08238', '5ucj', '5uc4'), ('P08254', '1b8y', '2usn'), ('P08254', '1ciz', '1hfs'), ('P08254', '1hfs', '1ciz'), ('P08254', '1sln', '2usn'), ('P08254', '1usn', '2usn'), ('P08254', '2d1o', '2usn'), ('P08254', '2usn', '1sln'), ('P08263', '1ydk', '4hj2'), ('P08263', '4hj2', '1ydk'), ('P08473', '1r1h', '1r1j'), ('P08473', '1r1j', '1r1h'), ('P08559', '3exe', '3exh'), ('P08559', '3exh', '3exe'), ('P08581', '2wgj', '3u6i'), ('P08581', '3q6w', '3zxz'), ('P08581', '3r7o', '3q6w'), ('P08581', '3u6h', '3u6i'), ('P08581', '3u6i', '3r7o'), ('P08581', '3zbx', '3u6i'), ('P08581', '3zc5', '3zcl'), ('P08581', '3zcl', '2wgj'), ('P08581', '3zxz', '3zc5'), ('P08709', '2b7d', '2bz6'), ('P08709', '2bz6', '5u6j'), ('P08709', '2flr', '4x8u'), ('P08709', '4ish', '4na9'), ('P08709', '4isi', '4x8u'), ('P08709', '4na9', '2flr'), ('P08709', '4x8u', '4ish'), ('P08709', '4x8v', '4ish'), ('P08709', '5l30', '4ish'), ('P08709', '5u6j', '4x8u'), ('P09211', '10gs', '3gss'), ('P09211', '1lbk', '5j41'), ('P09211', '2gss', '3gss'), ('P09211', '3gss', '5j41'), ('P09211', '5j41', '2gss'), ('P09237', '1mmq', '1mmr'), ('P09237', '1mmr', '1mmq'), ('P09382', '3oy8', '3oyw'), ('P09382', '3oyw', '3oy8'), ('P09455', '5ha1', '5hbs'), ('P09455', '5hbs', '5ha1'), ('P09464', '1lke', '1n0s'), ('P09464', '1lnm', '1n0s'), ('P09464', '1n0s', '1lnm'), ('P09874', '2rd6', '3gjw'), ('P09874', '3gjw', '5xsr'), ('P09874', '3l3l', '2rd6'), ('P09874', '3l3m', '5xsr'), ('P09874', '4und', '5xsr'), ('P09874', '4zzz', '2rd6'), ('P09874', '5xsr', '4und'), ('P09874', '6bhv', '4und'), ('P09958', '4omc', '4ryd'), ('P09958', '4ryd', '6eqx'), ('P09958', '6eqv', '6eqw'), ('P09958', '6eqw', '6eqv'), ('P09958', '6eqx', '4omc'), ('P09960', '2r59', '4l2l'), ('P09960', '3b7r', '2r59'), ('P09960', '3b7u', '3b7r'), ('P09960', '3fh7', '3b7r'), ('P09960', '4l2l', '3fh7'), ('P0A4Z6', '2xb8', '3n76'), ('P0A4Z6', '3n76', '3n7a'), ('P0A4Z6', '3n7a', '4ciw'), ('P0A4Z6', '3n86', '4b6o'), ('P0A4Z6', '3n8k', '3n76'), ('P0A4Z6', '4b6o', '2xb8'), ('P0A4Z6', '4b6p', '2xb8'), ('P0A4Z6', '4ciw', '3n76'), ('P0A4Z6', '4kiu', '3n86'), ('P0A538', '1g2o', '1n3i'), ('P0A538', '1n3i', '1g2o'), ('P0A546', '4km0', '4km2'), ('P0A546', '4km2', '4km0'), ('P0A5R0', '3cow', '3iod'), ('P0A5R0', '3coy', '3coz'), ('P0A5R0', '3coz', '3ime'), ('P0A5R0', '3imc', '3ioc'), ('P0A5R0', '3ime', '3ioc'), ('P0A5R0', '3iob', '3iod'), ('P0A5R0', '3ioc', '3iod'), ('P0A5R0', '3iod', '3coz'), ('P0A5R0', '3ioe', '3isj'), ('P0A5R0', '3isj', '3ioc'), ('P0A6D3', '1x8r', '1x8t'), ('P0A6D3', '1x8t', '1x8r'), ('P0A6D3', '2pq9', '1x8t'), ('P0A6I6', '6b7a', '6ckw'), ('P0A6I6', '6b7b', '6chp'), ('P0A6I6', '6chp', '6ckw'), ('P0A6I6', '6ckw', '6chp'), ('P0A6Y8', '4ezz', '4ezr'), ('P0A715', '1g7v', '1phw'), ('P0A715', '1phw', '1g7v'), ('P0A720', '4tmk', '5tmp'), ('P0A720', '5tmp', '4tmk'), ('P0A731', '1egh', '1ik4'), ('P0A731', '1ik4', '1s89'), ('P0A731', '1s89', '1egh'), ('P0A786', '1d09', '2fzk'), ('P0A786', '2fzc', '2fzk'), ('P0A786', '2fzk', '2h3e'), ('P0A786', '2h3e', '2fzc'), ('P0A884', '1f4e', '1f4f'), ('P0A884', '1f4f', '1f4e'), ('P0A884', '1f4g', '1f4f'), ('P0A8M3', '4hwo', '4hws'), ('P0A8M3', '4hwp', '4hws'), ('P0A8M3', '4hws', '4hwo'), ('P0A953', '2vb8', '2vba'), ('P0A953', '2vba', '2vb8'), ('P0A988', '4mjp', '4n9a'), ('P0A988', '4n9a', '4mjp'), ('P0ABF6', '1ctt', '1ctu'), ('P0ABF6', '1ctu', '1ctt'), ('P0ABP8', '1a69', '1k9s'), ('P0ABP8', '1k9s', '1a69'), ('P0ABQ4', '1dhi', '1dhj'), ('P0ABQ4', '1dhj', '1dhi'), ('P0ABQ4', '2drc', '1dhj'), ('P0AC14', '5u0w', '5u12'), ('P0AC14', '5u0y', '5u11'), ('P0AC14', '5u0z', '5v79'), ('P0AC14', '5u11', '5u0w'), ('P0AC14', '5u12', '5v7a'), ('P0AC14', '5u13', '5u0w'), ('P0AC14', '5u14', '5u11'), ('P0AC14', '5v79', '5u0y'), ('P0AC14', '5v7a', '5u0z'), ('P0AFG9', '3lpl', '3lq2'), ('P0AFG9', '3lq2', '3lpl'), ('P0C6F2', '3th9', '4mc1'), ('P0C6F2', '3vfa', '4mc9'), ('P0C6F2', '4i8w', '4qgi'), ('P0C6F2', '4i8z', '4mc2'), ('P0C6F2', '4mc1', '4mc9'), ('P0C6F2', '4mc2', '3th9'), ('P0C6F2', '4mc6', '4mc9'), ('P0C6F2', '4mc9', '3th9'), ('P0C6F2', '4qgi', '4i8w'), ('P0DMV8', '5aqz', '5mks'), ('P0DMV8', '5mkr', '5aqz'), ('P0DMV8', '5mks', '6fhk'), ('P0DMV8', '6fhk', '5aqz'), ('P0DOX7', '6msy', '6mub'), ('P0DOX7', '6mu3', '6msy'), ('P0DOX7', '6mub', '6msy'), ('P10153', '1hi3', '1hi5'), ('P10153', '1hi4', '1hi3'), ('P10153', '1hi5', '1hi4'), ('P10153', '5e13', '1hi4'), ('P10253', '5nn5', '5nn6'), ('P10253', '5nn6', '5nn5'), ('P10275', '1e3g', '3b5r'), ('P10275', '1xow', '3b66'), ('P10275', '1z95', '3b65'), ('P10275', '2ax9', '3b5r'), ('P10275', '3b5r', '1z95'), ('P10275', '3b66', '3b67'), ('P10275', '3b67', '1z95'), ('P10275', '3b68', '3b65'), ('P10275', '5cj6', '1xow'), ('P10415', '4ieh', '6gl8'), ('P10415', '4lvt', '6gl8'), ('P10415', '4lxd', '6gl8'), ('P10415', '6gl8', '4ieh'), ('P10845', '3c88', '3c89'), ('P10845', '3c89', '3c88'), ('P11021', '3ldp', '5f2r'), ('P11021', '5evz', '5ey4'), ('P11021', '5exw', '5f2r'), ('P11021', '5ey4', '3ldp'), ('P11021', '5f1x', '3ldp'), ('P11021', '5f2r', '3ldp'), ('P11142', '3ldq', '3m3z'), ('P11142', '3m3z', '3ldq'), ('P11309', '1xws', '4k0y'), ('P11309', '2c3i', '3bgq'), ('P11309', '2xj1', '2xj2'), ('P11309', '2xj2', '3bgz'), ('P11309', '3bgq', '3jya'), ('P11309', '3bgz', '2c3i'), ('P11309', '3jy0', '3bgz'), ('P11309', '3jya', '4k18'), ('P11309', '4k0y', '1xws'), ('P11309', '4k18', '2c3i'), ('P11362', '4rwj', '5am6'), ('P11362', '5am6', '4rwj'), ('P11362', '5am7', '5am6'), ('P11444', '3uxk', '3uxl'), ('P11444', '3uxl', '3uxk'), ('P11444', '4fp1', '4m6u'), ('P11444', '4m6u', '4fp1'), ('P11473', '1ie9', '1s19'), ('P11473', '1s19', '1ie9'), ('P11588', '1qy1', '1qy2'), ('P11588', '1qy2', '1qy1'), ('P11838', '1epo', '3wz7'), ('P11838', '1gvw', '3pww'), ('P11838', '1gvx', '3wz7'), ('P11838', '2v00', '1epo'), ('P11838', '3prs', '1epo'), ('P11838', '3pww', '2v00'), ('P11838', '3uri', '3wz8'), ('P11838', '3wz6', '1gvx'), ('P11838', '3wz7', '2v00'), ('P11838', '3wz8', '1gvw'), ('P12497', '4ahr', '4ahu'), ('P12497', '4ahs', '4ahr'), ('P12497', '4ahu', '4ahs'), ('P12499', '1hxw', '4ejl'), ('P12499', '1pro', '4ej8'), ('P12499', '4ej8', '1pro'), ('P12499', '4ejl', '1pro'), ('P12694', '1ols', '1olx'), ('P12694', '1olu', '1olx'), ('P12694', '1olx', '1v16'), ('P12694', '1v11', '1ols'), ('P12694', '1v16', '1olu'), ('P12694', '1v1m', '1v11'), ('P12821', '2oc2', '3bkl'), ('P12821', '2xyd', '6en5'), ('P12821', '3bkk', '6f9u'), ('P12821', '3bkl', '3nxq'), ('P12821', '3l3n', '4ca5'), ('P12821', '3nxq', '3bkl'), ('P12821', '4ca5', '3bkl'), ('P12821', '4ca6', '6f9u'), ('P12821', '6en5', '6f9u'), ('P12821', '6f9u', '3nxq'), ('P13009', '6bdy', '6bm6'), ('P13009', '6bm5', '6bdy'), ('P13009', '6bm6', '6bm5'), ('P13053', '1rjk', '2o4j'), ('P13053', '2o4j', '2o4r'), ('P13053', '2o4r', '2o4j'), ('P13482', '2jf4', '2jjb'), ('P13482', '2jg0', '2jjb'), ('P13482', '2jjb', '2jg0'), ('P13491', '4i8x', '4i9u'), ('P13491', '4i9h', '4i9u'), ('P13491', '4i9u', '4i8x'), ('P13631', '1fcx', '1fcz'), ('P13631', '1fcy', '1fcz'), ('P13631', '1fcz', '1fd0'), ('P13631', '1fd0', '1fcz'), ('P14061', '1i5r', '3hb4'), ('P14061', '3hb4', '1i5r'), ('P14174', '4wrb', '6cbf'), ('P14174', '5hvs', '6cbg'), ('P14174', '5hvt', '5hvs'), ('P14174', '5j7q', '4wrb'), ('P14174', '6b1k', '6cbf'), ('P14174', '6cbf', '6b1k'), ('P14174', '6cbg', '6b1k'), ('P14207', '4kmz', '4kn1'), ('P14207', '4kn0', '4kn1'), ('P14207', '4kn1', '4kn0'), ('P14324', '1yq7', '4pvy'), ('P14324', '2f94', '4pvy'), ('P14324', '2f9k', '1yq7'), ('P14324', '4pvx', '2f94'), ('P14324', '4pvy', '4pvx'), ('P14324', '5ja0', '2f94'), ('P14751', '1nw5', '1nw7'), ('P14751', '1nw7', '1nw5'), ('P14769', '1gwv', '1o7o'), ('P14769', '1o7o', '1gwv'), ('P15090', '2hnx', '5d45'), ('P15090', '5d45', '5y13'), ('P15090', '5d47', '5y13'), ('P15090', '5d48', '5d45'), ('P15090', '5edb', '2hnx'), ('P15090', '5edc', '2hnx'), ('P15090', '5hz6', '5y12'), ('P15090', '5hz8', '5edc'), ('P15090', '5y12', '5d47'), ('P15090', '5y13', '5y12'), ('P15207', '1i37', '3g0w'), ('P15207', '2ihq', '1i37'), ('P15207', '3g0w', '2ihq'), ('P15379', '4mre', '4mrg'), ('P15379', '4mrg', '4np2'), ('P15379', '4np2', '4mrg'), ('P15379', '4np3', '4np2'), ('P15917', '1yqy', '4dv8'), ('P15917', '4dv8', '1yqy'), ('P16088', '1fiv', '2hah'), ('P16088', '2hah', '1fiv'), ('P16404', '1ax0', '3n35'), ('P16404', '3n35', '1ax0'), ('P16932', '1m0n', '1zc9'), ('P16932', '1m0o', '1m0q'), ('P16932', '1m0q', '1zc9'), ('P16932', '1zc9', '1m0n'), ('P17050', '4do4', '4do5'), ('P17050', '4do5', '4do4'), ('P17612', '3agl', '5izj'), ('P17612', '4uj1', '4ujb'), ('P17612', '4uj2', '4uj1'), ('P17612', '4uja', '4uj2'), ('P17612', '4ujb', '3agl'), ('P17612', '5izf', '4uj2'), ('P17612', '5izj', '4uja'), ('P17752', '3hf8', '3hfb'), ('P17752', '3hfb', '3hf8'), ('P17931', '1kjr', '3t1m'), ('P17931', '3t1m', '6eol'), ('P17931', '5e89', '6eog'), ('P17931', '5h9r', '5e89'), ('P17931', '6eog', '1kjr'), ('P17931', '6eol', '5h9r'), ('P18031', '1bzc', '1ecv'), ('P18031', '1bzj', '1c84'), ('P18031', '1c83', '1bzc'), ('P18031', '1c84', '1bzc'), ('P18031', '1c86', '1bzc'), ('P18031', '1c87', '1ecv'), ('P18031', '1c88', '1bzj'), ('P18031', '1ecv', '1bzc'), ('P18031', '1g7f', '1ecv'), ('P18031', '1g7g', '1c83'), ('P18654', '3ubd', '4gue'), ('P18654', '4gue', '3ubd'), ('P18670', '1ugx', '1ws4'), ('P18670', '1ws4', '1ugx'), ('P19491', '1ftm', '1my4'), ('P19491', '1my4', '1p1o'), ('P19491', '1p1n', '1wvj'), ('P19491', '1p1o', '1syh'), ('P19491', '1p1q', '1p1o'), ('P19491', '1syh', '1syi'), ('P19491', '1syi', '1p1q'), ('P19491', '1wvj', '1p1o'), ('P19491', '1xhy', '1syi'), ('P19491', '2al5', '1syi'), ('P19492', '3dln', '3rt8'), ('P19492', '3dp4', '4f39'), ('P19492', '3rt8', '3dln'), ('P19492', '4f39', '3rt8'), ('P19493', '3fas', '3fat'), ('P19493', '3fat', '3fas'), ('P19793', '3ozj', '4m8e'), ('P19793', '4k4j', '4m8e'), ('P19793', '4k6i', '3ozj'), ('P19793', '4m8e', '4pp5'), ('P19793', '4m8h', '1fm9'), ('P19793', '4poj', '4poh'), ('P19793', '4pp3', '1fm9'), ('P19793', '4pp5', '4poh'), ('P19812', '3nik', '3nim'), ('P19812', '3nim', '3nik'), ('P19971', '1uou', '2wk6'), ('P19971', '2wk6', '1uou'), ('P20371', '1eoc', '2buv'), ('P20371', '2buv', '1eoc'), ('P20701', '1rd4', '3f78'), ('P20701', '3f78', '1rd4'), ('P20906', '3f6e', '3fzn'), ('P20906', '3fzn', '3f6e'), ('P21675', '5i1q', '5mg2'), ('P21675', '5i29', '5i1q'), ('P21675', '5mg2', '5i1q'), ('P21836', '1n5r', '2whp'), ('P21836', '1q84', '4ara'), ('P21836', '2ha2', '2ha3'), ('P21836', '2ha3', '1n5r'), ('P21836', '2ha6', '2ha3'), ('P21836', '2whp', '2ha3'), ('P21836', '4ara', '2ha6'), ('P21836', '4arb', '1q84'), ('P21836', '5ehq', '1n5r'), ('P22102', '1njs', '4ew2'), ('P22102', '4ew2', '4ew3'), ('P22102', '4ew3', '1njs'), ('P22303', '4m0e', '4m0f'), ('P22303', '4m0f', '4m0e'), ('P22392', '3bbb', '3bbf'), ('P22392', '3bbf', '3bbb'), ('P22498', '1uwt', '1uwu'), ('P22498', '1uwu', '1uwt'), ('P22498', '2ceq', '2cer'), ('P22498', '2cer', '2ceq'), ('P22629', '1df8', '1swr'), ('P22629', '1sld', '3rdo'), ('P22629', '1srg', '3wzn'), ('P22629', '1str', '1df8'), ('P22629', '1swg', '1swr'), ('P22629', '1swr', '1srg'), ('P22629', '2izl', '3wzn'), ('P22629', '3rdo', '1sld'), ('P22629', '3rdq', '3wzn'), ('P22629', '3wzn', '1srg'), ('P22734', '3hvi', '4p58'), ('P22734', '3hvj', '3u81'), ('P22734', '3oe4', '5k03'), ('P22734', '3oe5', '4p58'), ('P22734', '3ozr', '5k03'), ('P22734', '3ozs', '3oe4'), ('P22734', '3ozt', '3ozs'), ('P22734', '3u81', '5k03'), ('P22734', '4p58', '5k03'), ('P22734', '5k03', '3hvj'), ('P22756', '1vso', '4dld'), ('P22756', '2f34', '3gba'), ('P22756', '2f35', '3gbb'), ('P22756', '2pbw', '4dld'), ('P22756', '2wky', '3s2v'), ('P22756', '3gba', '2wky'), ('P22756', '3gbb', '2f35'), ('P22756', '3s2v', '2wky'), ('P22756', '4dld', '2f35'), ('P22756', '4e0x', '2pbw'), ('P22887', '1s5z', '4cp5'), ('P22887', '4cp5', '1s5z'), ('P22894', '1jao', '1zs0'), ('P22894', '1jaq', '3tt4'), ('P22894', '1zs0', '1jaq'), ('P22894', '1zvx', '1jao'), ('P22894', '3tt4', '1zvx'), ('P23458', '4e4l', '4ivb'), ('P23458', '4e4n', '4ivc'), ('P23458', '4e5w', '4ivd'), ('P23458', '4ehz', '4ei4'), ('P23458', '4ei4', '4ivd'), ('P23458', '4fk6', '4ehz'), ('P23458', '4i5c', '4ivc'), ('P23458', '4ivb', '4e4n'), ('P23458', '4ivc', '4i5c'), ('P23458', '4ivd', '4fk6'), ('P23616', '4bt3', '4bt5'), ('P23616', '4bt4', '4bt5'), ('P23616', '4bt5', '4bt3'), ('P23946', '1t31', '5yjm'), ('P23946', '3n7o', '5yjm'), ('P23946', '5yjm', '1t31'), ('P24182', '2v58', '2v59'), ('P24182', '2v59', '3rv4'), ('P24182', '3rv4', '2v59'), ('P24247', '1jys', '1nc3'), ('P24247', '1nc1', '1y6q'), ('P24247', '1nc3', '1jys'), ('P24247', '1y6q', '1y6r'), ('P24247', '1y6r', '1jys'), ('P24941', '1b38', '1e1x'), ('P24941', '1e1v', '1pxo'), ('P24941', '1e1x', '1h1s'), ('P24941', '1h1s', '1pxp'), ('P24941', '1jsv', '1pxp'), ('P24941', '1pxn', '1jsv'), ('P24941', '1pxo', '1pxn'), ('P24941', '1pxp', '1jsv'), ('P24941', '2exm', '1pxo'), ('P24941', '2fvd', '2exm'), ('P25440', '2ydw', '2yek'), ('P25440', '2yek', '2ydw'), ('P25440', '4mr6', '4uyf'), ('P25440', '4qev', '4uyf'), ('P25440', '4qew', '4uyf'), ('P25440', '4uyf', '4mr6'), ('P25774', '2f1g', '3n3g'), ('P25774', '2hhn', '3n3g'), ('P25774', '3n3g', '2f1g'), ('P26281', '1ex8', '4f7v'), ('P26281', '4f7v', '1ex8'), ('P26514', '1od8', '1v0k'), ('P26514', '1v0k', '1od8'), ('P26514', '1v0l', '1od8'), ('P26662', '3p8n', '3p8o'), ('P26662', '3p8o', '3p8n'), ('P26663', '1nhu', '3mf5'), ('P26663', '1os5', '1nhu'), ('P26663', '3cj2', '1nhu'), ('P26663', '3cj5', '3cj4'), ('P26918', '2gkl', '3iog'), ('P26918', '3iof', '3iog'), ('P26918', '3iog', '2gkl'), ('P27487', '1n1m', '2ole'), ('P27487', '2oag', '4lko'), ('P27487', '2ole', '4jh0'), ('P27487', '3sww', '4lko'), ('P27487', '4jh0', '2oag'), ('P27487', '4lko', '2oag'), ('P27487', '5kby', '3sww'), ('P27694', '4luz', '4o0a'), ('P27694', '4o0a', '4r4c'), ('P27694', '4r4c', '4r4t'), ('P27694', '4r4i', '4r4t'), ('P27694', '4r4o', '4r4i'), ('P27694', '4r4t', '4o0a'), ('P27694', '5e7n', '4r4o'), ('P28482', '2ojg', '3i5z'), ('P28482', '2ojj', '1pme'), ('P28482', '3i5z', '2ojg'), ('P28482', '3i60', '2ojg'), ('P28482', '4qp2', '2ojg'), ('P28523', '1m2p', '1zog'), ('P28523', '1m2q', '2oxd'), ('P28523', '1m2r', '1zog'), ('P28523', '1om1', '2oxy'), ('P28523', '1zoe', '1om1'), ('P28523', '1zog', '1m2p'), ('P28523', '1zoh', '2oxy'), ('P28523', '2oxd', '1om1'), ('P28523', '2oxx', '1om1'), ('P28523', '2oxy', '1om1'), ('P28720', '1enu', '1k4h'), ('P28720', '1f3e', '1k4h'), ('P28720', '1k4g', '2z1w'), ('P28720', '1k4h', '1f3e'), ('P28720', '1q65', '1s39'), ('P28720', '1r5y', '1s39'), ('P28720', '1s38', '1enu'), ('P28720', '1s39', '2z1w'), ('P28720', '2qzr', '1k4h'), ('P28720', '2z1w', '1q65'), ('P29029', '2uy3', '2uy5'), ('P29029', '2uy4', '2uy3'), ('P29029', '2uy5', '2uy4'), ('P29317', '5i9x', '5i9z'), ('P29317', '5i9y', '5ia1'), ('P29317', '5i9z', '5ia2'), ('P29317', '5ia0', '5i9y'), ('P29317', '5ia1', '5njz'), ('P29317', '5ia2', '5njz'), ('P29317', '5ia3', '5i9y'), ('P29317', '5ia4', '5i9x'), ('P29317', '5ia5', '5i9y'), ('P29317', '5njz', '5ia3'), ('P29375', '5ivc', '5ivv'), ('P29375', '5ive', '5ivv'), ('P29375', '5ivv', '5ivc'), ('P29375', '5ivy', '5ivv'), ('P29375', '6dq4', '5ive'), ('P29498', '1vyf', '1vyg'), ('P29498', '1vyg', '1vyf'), ('P29597', '3nyx', '4gj2'), ('P29597', '4gfo', '4gii'), ('P29597', '4gih', '4gj2'), ('P29597', '4gii', '4gfo'), ('P29597', '4gj2', '4gj3'), ('P29597', '4gj3', '4gfo'), ('P29597', '4wov', '3nyx'), ('P29597', '5wal', '4gii'), ('P29724', '2fqw', '2fqx'), ('P29724', '2fqx', '2fqw'), ('P29724', '2fqy', '2fqw'), ('P29736', '1e6q', '1e6s'), ('P29736', '1e6s', '1e6q'), ('P30044', '4k7i', '4k7o'), ('P30044', '4k7n', '4k7i'), ('P30044', '4k7o', '4k7i'), ('P30044', '4mmm', '4k7o'), ('P30113', '2c80', '2ca8'), ('P30113', '2ca8', '2c80'), ('P30291', '5vc3', '5vc4'), ('P30291', '5vc4', '5vd2'), ('P30291', '5vd2', '5vc3'), ('P30967', '3tk2', '4jpx'), ('P30967', '4jpx', '4jpy'), ('P30967', '4jpy', '4jpx'), ('P31151', '2wor', '2wos'), ('P31151', '2wos', '2wor'), ('P31947', '3iqu', '5btv'), ('P31947', '5btv', '3iqu'), ('P31992', '1gyx', '1gyy'), ('P31992', '1gyy', '1gyx'), ('P32890', '1jqy', '1pzi'), ('P32890', '1pzi', '1jqy'), ('P33038', '3upk', '3v4t'), ('P33038', '3v4t', '3upk'), ('P33981', '3hmp', '5n93'), ('P33981', '5mrb', '5n93'), ('P33981', '5n84', '5n93'), ('P33981', '5n93', '5n84'), ('P35030', '1h4w', '5tp0'), ('P35030', '5tp0', '1h4w'), ('P35031', '1utj', '1utl'), ('P35031', '1utl', '1utm'), ('P35031', '1utm', '1utj'), ('P35120', '4pow', '4pp0'), ('P35120', '4pox', '5ota'), ('P35120', '4pp0', '5ito'), ('P35120', '5ito', '5ota'), ('P35120', '5itp', '5ito'), ('P35120', '5ot8', '5ota'), ('P35120', '5ot9', '4pp0'), ('P35120', '5ota', '4pow'), ('P35120', '5otc', '5ot8'), ('P35439', '1pb8', '1y1z'), ('P35439', '1pb9', '1y1z'), ('P35439', '1pbq', '5vih'), ('P35439', '1y1z', '5vih'), ('P35439', '1y20', '5u8c'), ('P35439', '4kfq', '5u8c'), ('P35439', '5dex', '1pb8'), ('P35439', '5u8c', '4kfq'), ('P35439', '5vih', '1pbq'), ('P35439', '5vij', '5dex'), ('P35505', '1hyo', '2hzy'), ('P35505', '2hzy', '1hyo'), ('P35790', '4cg8', '4da5'), ('P35790', '4cg9', '4cga'), ('P35790', '4cga', '5eqe'), ('P35790', '4da5', '4cg9'), ('P35790', '5afv', '4da5'), ('P35790', '5eqe', '4cg9'), ('P35790', '5eqp', '4br3'), ('P35790', '5eqy', '3zm9'), ('P35963', '1k6c', '1t7j'), ('P35963', '1k6p', '1k6v'), ('P35963', '1k6t', '1k6c'), ('P35963', '1k6v', '1k6c'), ('P35963', '1t7j', '1k6p'), ('P35968', '2qu6', '3vhk'), ('P35968', '3vhk', '4asd'), ('P35968', '4ag8', '2qu6'), ('P35968', '4agc', '3vhk'), ('P35968', '4asd', '4agc'), ('P35968', '4ase', '3vhk'), ('P36186', '4bi6', '4bi7'), ('P36186', '4bi7', '4bi6'), ('P36639', '4c9x', '6f20'), ('P36639', '5ant', '6f20'), ('P36639', '5anu', '5fsn'), ('P36639', '5anv', '4c9x'), ('P36639', '5fsn', '5anu'), ('P36639', '5fso', '6f20'), ('P36639', '6f20', '5ant'), ('P37231', '2i4j', '4a4v'), ('P37231', '2i4z', '2yfe'), ('P37231', '2p4y', '4a4w'), ('P37231', '2yfe', '2p4y'), ('P37231', '3b1m', '2yfe'), ('P37231', '3fur', '2i4z'), ('P37231', '3u9q', '4r06'), ('P37231', '4a4v', '2yfe'), ('P37231', '4a4w', '2i4z'), ('P37231', '4r06', '4a4w'), ('P38998', '2qrk', '2qrl'), ('P38998', '2qrl', '2qrk'), ('P39086', '3fuz', '3fvk'), ('P39086', '3fv1', '3fuz'), ('P39086', '3fv2', '3fv1'), ('P39086', '3fvk', '3fvn'), ('P39086', '3fvn', '3fv1'), ('P39900', '1rmz', '3f1a'), ('P39900', '2hu6', '3ehx'), ('P39900', '3ehx', '2hu6'), ('P39900', '3ehy', '3f18'), ('P39900', '3f15', '3f19'), ('P39900', '3f16', '3f1a'), ('P39900', '3f17', '3f15'), ('P39900', '3f18', '1rmz'), ('P39900', '3f19', '3f18'), ('P39900', '3f1a', '2hu6'), ('P42260', '2xxr', '2xxt'), ('P42260', '2xxt', '2xxr'), ('P42260', '2xxx', '2xxr'), ('P42264', '3s9e', '4g8n'), ('P42264', '4g8n', '4nwc'), ('P42264', '4nwc', '4g8n'), ('P42336', '5dxt', '5uk8'), ('P42336', '5uk8', '5dxt'), ('P42530', '2vmc', '2vmd'), ('P42530', '2vmd', '2vmc'), ('P43166', '3mdz', '6h36'), ('P43166', '3ml5', '6h38'), ('P43166', '6h36', '6h38'), ('P43166', '6h37', '3mdz'), ('P43166', '6h38', '3ml5'), ('P43405', '3fqe', '4fl2'), ('P43405', '4fl1', '4fl2'), ('P43405', '4fl2', '3fqe'), ('P43490', '2gvj', '5upf'), ('P43490', '4n9c', '5upf'), ('P43490', '5upe', '4n9c'), ('P43490', '5upf', '2gvj'), ('P43619', '3c2f', '3c2r'), ('P43619', '3c2o', '3c2r'), ('P43619', '3c2r', '3c2o'), ('P44539', '1f73', '1f74'), ('P44539', '1f74', '1f73'), ('P44542', '2cex', '3b50'), ('P44542', '3b50', '2cex'), ('P45446', '1n4h', '1nq7'), ('P45446', '1nq7', '1n4h'), ('P45452', '2d1n', '3ljz'), ('P45452', '3kek', '2d1n'), ('P45452', '3ljz', '3tvc'), ('P45452', '3tvc', '2d1n'), ('P45452', '456c', '2d1n'), ('P45452', '4l19', '3tvc'), ('P46925', '1lee', '1lf2'), ('P46925', '1lf2', '1lee'), ('P47205', '2ves', '4lch'), ('P47205', '4lch', '2ves'), ('P47205', '5drr', '4lch'), ('P47228', '1kmy', '1lgt'), ('P47228', '1lgt', '1kmy'), ('P47811', '4loo', '2ewa'), ('P48499', '1amk', '2vxn'), ('P48499', '2vxn', '1amk'), ('P48825', '4iic', '4iif'), ('P48825', '4iid', '4iif'), ('P48825', '4iie', '4iif'), ('P48825', '4iif', '4iie'), ('P49336', '4crl', '4f6w'), ('P49336', '4f6u', '4crl'), ('P49336', '4f6w', '4crl'), ('P49610', '4az5', '4az6'), ('P49610', '4az6', '4azg'), ('P49610', '4azb', '4az5'), ('P49610', '4azc', '4azg'), ('P49610', '4azg', '4azi'), ('P49610', '4azi', '4azb'), ('P49773', '5i2e', '5wa9'), ('P49773', '5i2f', '5wa9'), ('P49773', '5ipc', '5wa8'), ('P49773', '5kly', '5ipc'), ('P49773', '5kma', '5wa8'), ('P49773', '5wa8', '5i2f'), ('P49773', '5wa9', '5kly'), ('P49841', '1q5k', '4acc'), ('P49841', '3i4b', '4acc'), ('P49841', '4acc', '1q5k'), ('P50053', '5wbm', '5wbo'), ('P50053', '5wbo', '5wbm'), ('P51449', '4ymq', '5ufr'), ('P51449', '5g45', '4ymq'), ('P51449', '5g46', '5vb5'), ('P51449', '5ufr', '5vb7'), ('P51449', '5vb5', '5g45'), ('P51449', '5vb6', '5ufr'), ('P51449', '5vb7', '5g46'), ('P51449', '6cn5', '5g45'), ('P52293', '4u54', '4u5s'), ('P52293', '4u5n', '4u5s'), ('P52293', '4u5o', '4u54'), ('P52293', '4u5s', '4u5o'), ('P52333', '3lxk', '5lwm'), ('P52333', '5lwm', '6gl9'), ('P52333', '6gl9', '6glb'), ('P52333', '6gla', '6gl9'), ('P52333', '6glb', '5lwm'), ('P52679', '4rpn', '4rpo'), ('P52679', '4rpo', '4rpn'), ('P52699', '2doo', '5ev8'), ('P52699', '5ev8', '5ewa'), ('P52699', '5ewa', '5ev8'), ('P52700', '2fu8', '5dpx'), ('P52700', '2qdt', '5evb'), ('P52700', '5dpx', '5evk'), ('P52700', '5evb', '5evk'), ('P52700', '5evd', '5evk'), ('P52700', '5evk', '5evd'), ('P52732', '2x2r', '5zo8'), ('P52732', '5zo8', '2x2r'), ('P53350', '3fvh', '4e67'), ('P53350', '4e67', '3fvh'), ('P53350', '4o6w', '4o9w'), ('P53350', '4o9w', '4o6w'), ('P53582', '4u1b', '4u69'), ('P53582', '4u69', '4u73'), ('P53582', '4u6c', '4u6z'), ('P53582', '4u6w', '4u69'), ('P53582', '4u6z', '4u70'), ('P53582', '4u70', '4u73'), ('P53582', '4u71', '4u6z'), ('P53582', '4u73', '4u1b'), ('P54760', '6fni', '6fnj'), ('P54760', '6fnj', '6fni'), ('P54818', '4ufh', '4ufi'), ('P54818', '4ufi', '4ufj'), ('P54818', '4ufj', '4ufh'), ('P54818', '4ufk', '4ufl'), ('P54818', '4ufl', '4ufm'), ('P54818', '4ufm', '4ufj'), ('P54829', '5ovr', '5ovx'), ('P54829', '5ovx', '5ovr'), ('P55055', '4rak', '5jy3'), ('P55055', '5jy3', '4rak'), ('P55072', '3hu3', '4ko8'), ('P55072', '4ko8', '3hu3'), ('P55201', '4uye', '5ov8'), ('P55201', '5eq1', '5mwh'), ('P55201', '5etb', '4uye'), ('P55201', '5mwh', '5ov8'), ('P55201', '5o5a', '5ov8'), ('P55201', '5ov8', '4uye'), ('P55201', '6ekq', '4uye'), ('P55212', '4n5d', '4nbk'), ('P55212', '4n6g', '4nbk'), ('P55212', '4n7m', '4n6g'), ('P55212', '4nbk', '4n6g'), ('P55212', '4nbl', '4n6g'), ('P55212', '4nbn', '4n6g'), ('P55859', '1a9q', '1lvu'), ('P55859', '1b8n', '1vfn'), ('P55859', '1b8o', '1lvu'), ('P55859', '1lvu', '1b8n'), ('P55859', '1v48', '1b8n'), ('P55859', '1vfn', '1a9q'), ('P55859', '3fuc', '1a9q'), ('P56109', '3c52', '3c56'), ('P56109', '3c56', '3c52'), ('P56221', '2std', '5std'), ('P56221', '3std', '2std'), ('P56221', '4std', '7std'), ('P56221', '5std', '3std'), ('P56221', '6std', '4std'), ('P56221', '7std', '4std'), ('P56658', '1ndv', '1o5r'), ('P56658', '1ndw', '1v7a'), ('P56658', '1ndy', '1o5r'), ('P56658', '1ndz', '1o5r'), ('P56658', '1o5r', '1ndw'), ('P56658', '1qxl', '1uml'), ('P56658', '1uml', '1v7a'), ('P56658', '1v7a', '1qxl'), ('P56658', '2e1w', '1ndv'), ('P56817', '1fkn', '2vkm'), ('P56817', '1m4h', '2qmg'), ('P56817', '2fdp', '2g94'), ('P56817', '2g94', '2fdp'), ('P56817', '2p4j', '3bug'), ('P56817', '2qmg', '2fdp'), ('P56817', '2vkm', '3buf'), ('P56817', '3bra', '2g94'), ('P56817', '3buf', '1m4h'), ('P56817', '3bug', '2p4j'), ('P58154', '1uv6', '3wtj'), ('P58154', '1uw6', '3wtm'), ('P58154', '3u8j', '3wtj'), ('P58154', '3u8k', '1uv6'), ('P58154', '3u8l', '3wtl'), ('P58154', '3u8n', '3wtj'), ('P58154', '3wtj', '3u8j'), ('P58154', '3wtl', '3wtm'), ('P58154', '3wtm', '3u8l'), ('P58154', '3wtn', '1uw6'), ('P59071', '1fv0', '1sv3'), ('P59071', '1jq8', '1fv0'), ('P59071', '1kpm', '1jq8'), ('P59071', '1q7a', '2arm'), ('P59071', '1sv3', '1q7a'), ('P59071', '2arm', '1kpm'), ('P59071', '3h1x', '1kpm'), ('P60045', '1oxr', '1td7'), ('P60045', '1td7', '1oxr'), ('P61823', '1afk', '1o0h'), ('P61823', '1afl', '1afk'), ('P61823', '1jn4', '1jvu'), ('P61823', '1jvu', '1rnm'), ('P61823', '1o0f', '1o0h'), ('P61823', '1o0h', '1afk'), ('P61823', '1o0m', '1qhc'), ('P61823', '1o0n', '1afk'), ('P61823', '1qhc', '1o0h'), ('P61823', '1rnm', '1afl'), ('P61964', '4ql1', '6dar'), ('P61964', '5m23', '5sxm'), ('P61964', '5m25', '5sxm'), ('P61964', '5sxm', '6dar'), ('P61964', '6d9x', '5m25'), ('P61964', '6dai', '6dak'), ('P61964', '6dak', '6dai'), ('P61964', '6dar', '6dai'), ('P62508', '2p7a', '2p7g'), ('P62508', '2p7g', '2p7z'), ('P62617', '2amt', '3elc'), ('P62617', '2gzl', '3elc'), ('P62617', '3elc', '2amt'), ('P62937', '5t9u', '6gjj'), ('P62937', '5t9w', '5ta4'), ('P62937', '5t9z', '6gjm'), ('P62937', '6gji', '5t9w'), ('P62937', '6gjj', '5ta4'), ('P62937', '6gjl', '6gjm'), ('P62937', '6gjm', '5t9z'), ('P62937', '6gjn', '5t9z'), ('P62942', '1d7i', '1fkf'), ('P62942', '1d7j', '1fkb'), ('P62942', '1fkg', '1d7j'), ('P62942', '1fkh', '1fkf'), ('P62942', '1fki', '1fkf'), ('P62942', '1j4r', '1fkg'), ('P62993', '3ov1', '3s8n'), ('P62993', '3ove', '3ov1'), ('P62993', '3s8l', '3s8o'), ('P62993', '3s8n', '3ov1'), ('P62993', '3s8o', '3ove'), ('P63086', '4qyy', '6cpw'), ('P63086', '6cpw', '4qyy'), ('P64012', '3zhx', '3zi0'), ('P64012', '3zi0', '3zhx'), ('P66034', '2c92', '2c97'), ('P66034', '2c94', '2c97'), ('P66034', '2c97', '2c94'), ('P66992', '3qqs', '3r88'), ('P66992', '3r88', '4m0r'), ('P66992', '3twp', '3uu1'), ('P66992', '3uu1', '3twp'), ('P66992', '4gkm', '3qqs'), ('P66992', '4ij1', '4owv'), ('P66992', '4m0r', '4gkm'), ('P66992', '4n8q', '3twp'), ('P66992', '4owm', '4m0r'), ('P66992', '4owv', '3twp'), ('P68400', '2zjw', '5h8e'), ('P68400', '3bqc', '5cu4'), ('P68400', '3h30', '3pe2'), ('P68400', '3pe1', '5cu4'), ('P68400', '3pe2', '3pe1'), ('P68400', '5cqu', '5cu4'), ('P68400', '5csp', '3h30'), ('P68400', '5cu4', '3pe1'), ('P68400', '5h8e', '5csp'), ('P69834', '5mrm', '5mro'), ('P69834', '5mro', '5mrp'), ('P69834', '5mrp', '5mro'), ('P71094', '2jke', '2zq0'), ('P71094', '2jkp', '2zq0'), ('P71094', '2zq0', '2jkp'), ('P71447', '1z4o', '2wf5'), ('P71447', '2wf5', '1z4o'), ('P76141', '4l4z', '4l51'), ('P76141', '4l50', '4l51'), ('P76141', '4l51', '4l50'), ('P76637', '1ec9', '1ecq'), ('P76637', '1ecq', '1ec9'), ('P78536', '2oi0', '3l0v'), ('P78536', '3b92', '3le9'), ('P78536', '3ewj', '3le9'), ('P78536', '3kmc', '3le9'), ('P78536', '3l0v', '3lea'), ('P78536', '3le9', '3ewj'), ('P78536', '3lea', '3le9'), ('P80188', '3dsz', '3tf6'), ('P80188', '3tf6', '3dsz'), ('P84887', '2hjb', '2q7q'), ('P84887', '2q7q', '2hjb'), ('P95607', '3i4y', '3i51'), ('P95607', '3i51', '3i4y'), ('P96257', '6h1u', '6h2t'), ('P96257', '6h2t', '6h1u'), ('P98170', '2vsl', '4j46'), ('P98170', '3cm2', '3hl5'), ('P98170', '3hl5', '4j48'), ('P98170', '4j44', '4j45'), ('P98170', '4j45', '4j44'), ('P98170', '4j46', '3hl5'), ('P98170', '4j47', '4j44'), ('P98170', '4j48', '3cm2'), ('P9WMC0', '5f08', '5j3l'), ('P9WMC0', '5f0f', '5j3l'), ('P9WMC0', '5j3l', '5f08'), ('P9WPQ5', '6cvf', '6czc'), ('P9WPQ5', '6czb', '6cze'), ('P9WPQ5', '6czc', '6czb'), ('P9WPQ5', '6cze', '6czc'), ('Q00972', '3tz0', '4h7q'), ('Q00972', '4dzy', '4h85'), ('Q00972', '4h7q', '4h85'), ('Q00972', '4h81', '4h7q'), ('Q00972', '4h85', '4dzy'), ('Q00987', '4erf', '4wt2'), ('Q00987', '4hbm', '4mdn'), ('Q00987', '4mdn', '4wt2'), ('Q00987', '4wt2', '4mdn'), ('Q00987', '4zyf', '4hbm'), ('Q01693', '1ft7', '3b3w'), ('Q01693', '1igb', '1ft7'), ('Q01693', '1txr', '1igb'), ('Q01693', '3b3c', '3b7i'), ('Q01693', '3b3s', '3b3c'), ('Q01693', '3b3w', '1txr'), ('Q01693', '3b7i', '1igb'), ('Q01693', '3vh9', '3b3w'), ('Q03111', '6hpw', '6ht1'), ('Q03111', '6ht1', '6hpw'), ('Q04609', '2xef', '2xei'), ('Q04609', '2xei', '4ngp'), ('Q04609', '2xej', '3sjf'), ('Q04609', '3iww', '3rbu'), ('Q04609', '3rbu', '2xef'), ('Q04609', '3sjf', '4ngn'), ('Q04609', '4ngm', '4ngp'), ('Q04609', '4ngn', '2xeg'), ('Q04609', '4ngp', '2xeg'), ('Q04631', '1o1s', '1qbq'), ('Q04631', '1qbq', '1o1s'), ('Q05097', '2wyf', '4a6s'), ('Q05097', '3zyf', '4ljh'), ('Q05097', '4a6s', '4ljh'), ('Q05097', '4ljh', '3zyf'), ('Q05097', '4lk7', '2wyf'), ('Q05127', '4ibb', '4ibf'), ('Q05127', '4ibc', '4ibg'), ('Q05127', '4ibd', '4ibc'), ('Q05127', '4ibe', '4ibd'), ('Q05127', '4ibf', '4ibe'), ('Q05127', '4ibg', '4ibe'), ('Q05127', '4ibi', '4ibc'), ('Q05127', '4ibj', '4ibb'), ('Q05127', '4ibk', '4ibc'), ('Q05397', '4gu6', '4kao'), ('Q05397', '4gu9', '4kao'), ('Q05397', '4k9y', '4kao'), ('Q05397', '4kao', '4k9y'), ('Q06135', '5o9o', '5o9y'), ('Q06135', '5o9p', '5o9y'), ('Q06135', '5o9q', '5o9r'), ('Q06135', '5o9r', '5o9o'), ('Q06135', '5o9y', '5o9q'), ('Q06135', '5oa2', '5oa6'), ('Q06135', '5oa6', '5o9p'), ('Q06GJ0', '3nsn', '3s6t'), ('Q06GJ0', '3ozp', '3nsn'), ('Q06GJ0', '3s6t', '3ozp'), ('Q06GJ0', '3vtr', '3wmc'), ('Q06GJ0', '3wmc', '3vtr'), ('Q07075', '4kx8', '4kxb'), ('Q07075', '4kxb', '4kx8'), ('Q07343', '1ro6', '3o56'), ('Q07343', '3o56', '5laq'), ('Q07343', '5laq', '3o56'), ('Q07817', '2yxj', '3spf'), ('Q07817', '3qkd', '3zln'), ('Q07817', '3spf', '4c5d'), ('Q07817', '3zk6', '3zlr'), ('Q07817', '3zln', '3spf'), ('Q07817', '3zlr', '4c52'), ('Q07817', '4c52', '3zlr'), ('Q07817', '4c5d', '3spf'), ('Q07820', '4hw3', '6b4u'), ('Q07820', '4zbf', '6b4l'), ('Q07820', '4zbi', '6fs0'), ('Q07820', '5vkc', '6fs1'), ('Q07820', '6b4l', '4zbi'), ('Q07820', '6b4u', '6fs1'), ('Q07820', '6fs0', '6b4u'), ('Q07820', '6fs1', '6b4u'), ('Q08638', '1oif', '2j75'), ('Q08638', '1uz1', '2j75'), ('Q08638', '1w3j', '1uz1'), ('Q08638', '2cbu', '2j75'), ('Q08638', '2cbv', '2j78'), ('Q08638', '2ces', '2j78'), ('Q08638', '2cet', '1oif'), ('Q08638', '2j75', '2cbu'), ('Q08638', '2j77', '2cet'), ('Q08638', '2j78', '1oif'), ('Q08881', '3miy', '3qgw'), ('Q08881', '3qgw', '4qd6'), ('Q08881', '3qgy', '4qd6'), ('Q08881', '4m0y', '4m13'), ('Q08881', '4m12', '3qgw'), ('Q08881', '4m13', '4m14'), ('Q08881', '4m14', '4qd6'), ('Q08881', '4qd6', '3qgw'), ('Q08881', '4rfm', '4m0y'), ('Q0A480', '4ks1', '4ks4'), ('Q0A480', '4ks4', '4ks1'), ('Q0ED31', '4dkp', '4dko'), ('Q0ED31', '4dkq', '4dkp'), ('Q0ED31', '4dkr', '4dkq'), ('Q0ED31', '4i54', '4dkp'), ('Q0P8Q4', '5ad1', '5tcy'), ('Q0P8Q4', '5tcy', '5ad1'), ('Q0PBL7', '3fj7', '3fjg'), ('Q0PBL7', '3fjg', '3fj7'), ('Q0T8Y8', '4xoc', '4xoe'), ('Q0T8Y8', '4xoe', '4xoc'), ('Q0TR53', '2j62', '2x0y'), ('Q0TR53', '2wb5', '2xpk'), ('Q0TR53', '2x0y', '2wb5'), ('Q0TR53', '2xpk', '2j62'), ('Q10714', '1j36', '4ca7'), ('Q10714', '1j37', '2x95'), ('Q10714', '2x8z', '1j37'), ('Q10714', '2x91', '4ca8'), ('Q10714', '2x95', '2x97'), ('Q10714', '2x96', '2x91'), ('Q10714', '2x97', '1j37'), ('Q10714', '2xhm', '2x91'), ('Q10714', '4ca7', '2x95'), ('Q10714', '4ca8', '1j36'), ('Q12051', '2e91', '2e92'), ('Q12051', '2e92', '2e91'), ('Q12051', '2e94', '2e91'), ('Q12852', '5cep', '5vo1'), ('Q12852', '5ceq', '5vo1'), ('Q12852', '5vo1', '5ceq'), ('Q13093', '5lz4', '5lz5'), ('Q13093', '5lz5', '5lz4'), ('Q13093', '5lz7', '5lz4'), ('Q13133', '3ipu', '3ipq'), ('Q13153', '4zji', '5ime'), ('Q13153', '5dey', '4zji'), ('Q13153', '5dfp', '4zji'), ('Q13153', '5ime', '5dfp'), ('Q13451', '4jfk', '4w9o'), ('Q13451', '4jfm', '4w9p'), ('Q13451', '4w9o', '4jfk'), ('Q13451', '4w9p', '4jfk'), ('Q13451', '5dit', '4w9p'), ('Q13526', '2xp7', '3ikd'), ('Q13526', '3ikd', '3ikg'), ('Q13526', '3ikg', '2xp7'), ('Q13627', '6eif', '6eis'), ('Q13627', '6eij', '6eis'), ('Q13627', '6eiq', '6eir'), ('Q13627', '6eir', '6eij'), ('Q13627', '6eis', '6eiq'), ('Q14145', '4xmb', '5x54'), ('Q14145', '5x54', '4xmb'), ('Q14397', '4bb9', '4ly9'), ('Q14416', '4xaq', '4xas'), ('Q14416', '4xas', '4xaq'), ('Q15119', '4mpn', '5j6a'), ('Q15119', '5j6a', '4mpn'), ('Q15370', '4b9k', '4w9d'), ('Q15370', '4bks', '4b9k'), ('Q15370', '4bkt', '4w9h'), ('Q15370', '4w9c', '4w9h'), ('Q15370', '4w9d', '4w9k'), ('Q15370', '4w9f', '4b9k'), ('Q15370', '4w9h', '4w9j'), ('Q15370', '4w9i', '4w9k'), ('Q15370', '4w9j', '4w9i'), ('Q15370', '4w9k', '4w9f'), ('Q15562', '5dq8', '5dqe'), ('Q15562', '5dqe', '5dq8'), ('Q16539', '1kv1', '1yqj'), ('Q16539', '1yqj', '1kv1'), ('Q16539', '2baj', '3e92'), ('Q16539', '2bak', '3d7z'), ('Q16539', '2bal', '3d83'), ('Q16539', '2yix', '2zb1'), ('Q16539', '2zb1', '2yix'), ('Q16539', '3d7z', '2zb1'), ('Q16539', '3d83', '2bak'), ('Q16539', '3e92', '1kv1'), ('Q16769', '2afw', '2afx'), ('Q16769', '2afx', '2afw'), ('Q16769', '3pbb', '2afx'), ('Q18R04', '2h6b', '3e5u'), ('Q18R04', '3e5u', '2h6b'), ('Q1W640', '3s0b', '3s0d'), ('Q1W640', '3s0d', '3s0b'), ('Q1W640', '3s0e', '3s0d'), ('Q24451', '1ps3', '3d51'), ('Q24451', '2f7o', '1ps3'), ('Q24451', '2f7p', '3d51'), ('Q24451', '3d4y', '3d50'), ('Q24451', '3d4z', '3d52'), ('Q24451', '3d50', '3ddf'), ('Q24451', '3d51', '1ps3'), ('Q24451', '3d52', '3d50'), ('Q24451', '3ddf', '3ddg'), ('Q24451', '3ddg', '3ddf'), ('Q26998', '1jlr', '1upf'), ('Q26998', '1upf', '1jlr'), ('Q2A1P5', '4hy1', '4hym'), ('Q2A1P5', '4hym', '4hy1'), ('Q2PS28', '2pwd', '2pwg'), ('Q2PS28', '2pwg', '2pwd'), ('Q38BV6', '2ptz', '2pu1'), ('Q38BV6', '2pu1', '2ptz'), ('Q396C9', '3b4p', '3juo'), ('Q396C9', '3juo', '3jup'), ('Q396C9', '3jup', '3juo'), ('Q41931', '5gj9', '5gja'), ('Q41931', '5gja', '5gj9'), ('Q42975', '3f5j', '3f5k'), ('Q42975', '3f5k', '3f5l'), ('Q42975', '3f5l', '3f5k'), ('Q460N5', '5o2d', '3q71'), ('Q46822', '1q54', '2vnp'), ('Q46822', '2vnp', '1q54'), ('Q48255', '2xd9', '4b6s'), ('Q48255', '2xda', '4b6r'), ('Q48255', '4b6r', '2xd9'), ('Q48255', '4b6s', '2xd9'), ('Q4W803', '3ahn', '3aho'), ('Q4W803', '3aho', '3ahn'), ('Q52L64', '2r1y', '2r23'), ('Q52L64', '2r23', '2r1y'), ('Q54727', '2vw1', '2vw2'), ('Q54727', '2vw2', '2vw1'), ('Q57ZL6', '4i71', '4i72'), ('Q57ZL6', '4i72', '4i74'), ('Q57ZL6', '4i74', '4i72'), ('Q58597', '4rrf', '4rrg'), ('Q58597', '4rrg', '4rrf'), ('Q58EU8', '1yei', '2c1p'), ('Q58EU8', '2c1p', '1yei'), ('Q58F21', '4flp', '4kcx'), ('Q58F21', '4kcx', '4flp'), ('Q5A4W8', '5n17', '5n18'), ('Q5A4W8', '5n18', '5n17'), ('Q5G940', '2glp', '3b7j'), ('Q5G940', '3b7j', '3ed0'), ('Q5G940', '3cf8', '3b7j'), ('Q5G940', '3ed0', '3b7j'), ('Q5M4H8', '5oxk', '6ghj'), ('Q5M4H8', '6ghj', '5oxk'), ('Q5RZ08', '2r3w', '6cdl'), ('Q5RZ08', '2r43', '6dil'), ('Q5RZ08', '5wlo', '6cdj'), ('Q5RZ08', '6cdj', '2r3t'), ('Q5RZ08', '6cdl', '2r43'), ('Q5RZ08', '6dif', '5wlo'), ('Q5RZ08', '6dil', '6dj1'), ('Q5RZ08', '6dj1', '2r43'), ('Q5SH52', '1wuq', '1wur'), ('Q5SH52', '1wur', '1wuq'), ('Q5SID9', '1odi', '1odj'), ('Q5SID9', '1odj', '1odi'), ('Q63226', '2v3u', '5cc2'), ('Q63226', '5cc2', '2v3u'), ('Q6DPL2', '3ckz', '3cl0'), ('Q6DPL2', '3cl0', '3ckz'), ('Q6G8R1', '4ai5', '4aia'), ('Q6G8R1', '4aia', '4ai5'), ('Q6N193', '5oei', '5oku'), ('Q6N193', '5oku', '5oei'), ('Q6P6C2', '4o61', '4oct'), ('Q6P6C2', '4oct', '4o61'), ('Q6PL18', '4qsu', '4tu4'), ('Q6PL18', '4qsv', '4tte'), ('Q6PL18', '4tt2', '4qsv'), ('Q6PL18', '4tte', '4tz2'), ('Q6PL18', '4tu4', '4qsu'), ('Q6PL18', '4tz2', '4qsu'), ('Q6PL18', '5a5q', '5a81'), ('Q6PL18', '5a81', '4tt2'), ('Q6R308', '3f7h', '3f7i'), ('Q6R308', '3f7i', '3gta'), ('Q6R308', '3gt9', '3f7i'), ('Q6R308', '3gta', '3f7h'), ('Q6T755', '3uj9', '3ujc'), ('Q6T755', '3ujc', '3uj9'), ('Q6T755', '3ujd', '3ujc'), ('Q6WVP6', '4q3t', '4q3u'), ('Q6WVP6', '4q3u', '4q3t'), ('Q6XEC0', '5qa8', '5qal'), ('Q6XEC0', '5qal', '5qay'), ('Q6XEC0', '5qay', '5qal'), ('Q70I53', '5g17', '5g1a'), ('Q70I53', '5g1a', '5g17'), ('Q72498', '3ao2', '3ao5'), ('Q72498', '3ao4', '3ovn'), ('Q72498', '3ao5', '3ao2'), ('Q72498', '3ovn', '3ao5'), ('Q75I93', '4qlk', '4qll'), ('Q75I93', '4qll', '4qlk'), ('Q76353', '3zso', '4ceb'), ('Q76353', '3zsq', '3zso'), ('Q76353', '3zsx', '3zt2'), ('Q76353', '3zsy', '4ceb'), ('Q76353', '3zt2', '4cgi'), ('Q76353', '3zt3', '4ceb'), ('Q76353', '4ceb', '4cig'), ('Q76353', '4cgi', '4cj4'), ('Q76353', '4cig', '3zsx'), ('Q76353', '4cj4', '3zsq'), ('Q7B8P6', '3qps', '3qqa'), ('Q7B8P6', '3qqa', '3qps'), ('Q7CX36', '3ip5', '3ip9'), ('Q7CX36', '3ip6', '3ip9'), ('Q7CX36', '3ip9', '3ip6'), ('Q7D2F4', '4ra1', '4zeb'), ('Q7D2F4', '4zeb', '4zei'), ('Q7D2F4', '4zec', '4zei'), ('Q7D2F4', '4zei', '4zec'), ('Q7D447', '5l9o', '5lom'), ('Q7D447', '5lom', '5l9o'), ('Q7D737', '2bes', '2bet'), ('Q7D737', '2bet', '2bes'), ('Q7D785', '3rv8', '3veh'), ('Q7D785', '3veh', '3rv8'), ('Q7DDU0', '4ipi', '4ipj'), ('Q7DDU0', '4ipj', '4ipi'), ('Q7SSI0', '3s54', '4kb9'), ('Q7SSI0', '4kb9', '3s54'), ('Q7ZCI0', '6dh1', '6dh2'), ('Q7ZCI0', '6dh2', '6dh1'), ('Q81R22', '4elf', '4elg'), ('Q81R22', '4elg', '4elf'), ('Q81R22', '4elh', '4elf'), ('Q86C09', '2i19', '3egt'), ('Q86C09', '3egt', '2i19'), ('Q86U86', '5fh7', '5ii2'), ('Q86U86', '5fh8', '5hrv'), ('Q86U86', '5hrv', '5hrw'), ('Q86U86', '5hrw', '5hrv'), ('Q86U86', '5hrx', '5fh7'), ('Q86U86', '5ii2', '5fh7'), ('Q86WV6', '4loi', '4qxo'), ('Q86WV6', '4qxo', '4ksy'), ('Q873X9', '1w9u', '1w9v'), ('Q873X9', '1w9v', '2iuz'), ('Q873X9', '2iuz', '1w9v'), ('Q89ZI2', '2j47', '2w4x'), ('Q89ZI2', '2j4g', '2w67'), ('Q89ZI2', '2jiw', '2w66'), ('Q89ZI2', '2vvn', '2xj7'), ('Q89ZI2', '2vvs', '2wca'), ('Q89ZI2', '2w4x', '2w67'), ('Q89ZI2', '2w66', '2j4g'), ('Q89ZI2', '2w67', '2jiw'), ('Q89ZI2', '2wca', '2jiw'), ('Q89ZI2', '2xj7', '2j47'), ('Q8A0N1', '2wvz', '2wzs'), ('Q8A0N1', '2wzs', '2wvz'), ('Q8A3I4', '2xib', '4j28'), ('Q8A3I4', '2xii', '4pcs'), ('Q8A3I4', '4j28', '4pee'), ('Q8A3I4', '4jfs', '2wvt'), ('Q8A3I4', '4pcs', '4jfs'), ('Q8A3I4', '4pee', '2wvt'), ('Q8AAK6', '2vjx', '2vot'), ('Q8AAK6', '2vl4', '2vjx'), ('Q8AAK6', '2vmf', '2vqt'), ('Q8AAK6', '2vo5', '2vl4'), ('Q8AAK6', '2vot', '2vjx'), ('Q8AAK6', '2vqt', '2vot'), ('Q8I3X4', '1nw4', '1q1g'), ('Q8I3X4', '1q1g', '1nw4'), ('Q8II92', '2y8c', '3t64'), ('Q8II92', '3t60', '3t64'), ('Q8II92', '3t64', '3t70'), ('Q8II92', '3t70', '2y8c'), ('Q8N1Q1', '3czv', '4hu1'), ('Q8N1Q1', '4hu1', '4knm'), ('Q8N1Q1', '4knm', '4qjx'), ('Q8N1Q1', '4knn', '4qjx'), ('Q8N1Q1', '4qjx', '4knm'), ('Q8Q3H0', '2cej', '4a6b'), ('Q8Q3H0', '2cen', '4a4q'), ('Q8Q3H0', '4a4q', '2cej'), ('Q8Q3H0', '4a6b', '4a6c'), ('Q8Q3H0', '4a6c', '5dgu'), ('Q8Q3H0', '5dgu', '4a4q'), ('Q8Q3H0', '5dgw', '5dgu'), ('Q8TEK3', '3qox', '3sr4'), ('Q8TEK3', '3sr4', '3qox'), ('Q8TEK3', '4ek9', '3qox'), ('Q8TF76', '6g34', '6g3a'), ('Q8TF76', '6g35', '6g3a'), ('Q8TF76', '6g36', '6g38'), ('Q8TF76', '6g37', '6g35'), ('Q8TF76', '6g38', '6g34'), ('Q8TF76', '6g39', '6g3a'), ('Q8TF76', '6g3a', '6g35'), ('Q8ULI3', '6dh6', '6dh8'), ('Q8ULI3', '6dh7', '6dh6'), ('Q8ULI3', '6dh8', '6dh7'), ('Q8VPB3', '2vpn', '2vpo'), ('Q8VPB3', '2vpo', '2vpn'), ('Q8WQX9', '5g2b', '5l8y'), ('Q8WQX9', '5g57', '5l8y'), ('Q8WQX9', '5g5v', '5l8c'), ('Q8WQX9', '5l8c', '5g57'), ('Q8WQX9', '5l8y', '5l8c'), ('Q8WSF8', '2byr', '2xys'), ('Q8WSF8', '2bys', '2wnj'), ('Q8WSF8', '2pgz', '2ymd'), ('Q8WSF8', '2wn9', '2wnj'), ('Q8WSF8', '2wnc', '2wn9'), ('Q8WSF8', '2wnj', '2wn9'), ('Q8WSF8', '2x00', '2wn9'), ('Q8WSF8', '2xys', '2x00'), ('Q8WSF8', '2xyt', '2wn9'), ('Q8WSF8', '2ymd', '2x00'), ('Q8WUI4', '3znr', '3zns'), ('Q8WUI4', '3zns', '3znr'), ('Q8WXF7', '4idn', '4ido'), ('Q8WXF7', '4ido', '4idn'), ('Q8XXK6', '2bt9', '4csd'), ('Q8XXK6', '4csd', '2bt9'), ('Q8Y8D7', '2i2c', '4dy6'), ('Q8Y8D7', '4dy6', '5dhu'), ('Q8Y8D7', '5dhu', '4dy6'), ('Q90EB9', '1izh', '1izi'), ('Q90EB9', '1izi', '1izh'), ('Q90JJ9', '4m8x', '4m8y'), ('Q90JJ9', '4m8y', '4m8x'), ('Q90K99', '3o99', '4djr'), ('Q90K99', '3o9a', '3o99'), ('Q90K99', '3o9d', '3o99'), ('Q90K99', '3o9e', '4djp'), ('Q90K99', '3o9i', '3o99'), ('Q90K99', '4djo', '3o9i'), ('Q90K99', '4djp', '3o9i'), ('Q90K99', '4djq', '3o9a'), ('Q90K99', '4djr', '3o9e'), ('Q92769', '4lxz', '5iwg'), ('Q92769', '4ly1', '5ix0'), ('Q92769', '5iwg', '4ly1'), ('Q92769', '5ix0', '4ly1'), ('Q92793', '4tqn', '5h85'), ('Q92793', '4yk0', '5mme'), ('Q92793', '5dbm', '5eng'), ('Q92793', '5eng', '5h85'), ('Q92793', '5ep7', '5i8g'), ('Q92793', '5h85', '5ep7'), ('Q92793', '5i8g', '5j0d'), ('Q92793', '5j0d', '5dbm'), ('Q92793', '5mme', '5dbm'), ('Q92793', '5mmg', '5h85'), ('Q92831', '5fe6', '5lvq'), ('Q92831', '5fe7', '5lvr'), ('Q92831', '5fe9', '5fe7'), ('Q92831', '5lvq', '5fe7'), ('Q92831', '5lvr', '5lvq'), ('Q92N37', '2reg', '2rin'), ('Q92N37', '2rin', '2reg'), ('Q92WC8', '2q88', '2q89'), ('Q92WC8', '2q89', '2q88'), ('Q939R9', '3str', '3sw8'), ('Q939R9', '3sw8', '3str'), ('Q93UV0', '2w8j', '2w8w'), ('Q93UV0', '2w8w', '2w8j'), ('Q95NY5', '3nhi', '3nht'), ('Q95NY5', '3nht', '3nhi'), ('Q96CA5', '2i3h', '2i3i'), ('Q96CA5', '2i3i', '3f7g'), ('Q96CA5', '3f7g', '3uw5'), ('Q96CA5', '3uw5', '2i3h'), ('Q980A5', '4rd0', '4rd3'), ('Q980A5', '4rd3', '4rd6'), ('Q980A5', '4rd6', '4rd3'), ('Q99640', '5vcv', '5vcy'), ('Q99640', '5vcw', '5vd1'), ('Q99640', '5vcy', '5vd0'), ('Q99640', '5vcz', '5vd3'), ('Q99640', '5vd0', '5vd1'), ('Q99640', '5vd1', '5vcy'), ('Q99640', '5vd3', '5vd0'), ('Q99814', '4ghi', '5tbm'), ('Q99814', '5tbm', '5ufp'), ('Q99814', '5ufp', '5tbm'), ('Q99AU2', '2d3u', '2d3z'), ('Q99AU2', '2d3z', '2d3u'), ('Q99QC1', '5k1d', '5k1f'), ('Q99QC1', '5k1f', '5k1d'), ('Q9AMP1', '3arp', '3arx'), ('Q9AMP1', '3arq', '3arw'), ('Q9AMP1', '3arw', '3arp'), ('Q9AMP1', '3arx', '3arp'), ('Q9BJF5', '3sxf', '3v5p'), ('Q9BJF5', '3t3u', '3v5t'), ('Q9BJF5', '3v51', '3v5t'), ('Q9BJF5', '3v5p', '3v51'), ('Q9BJF5', '3v5t', '3v5p'), ('Q9BY41', '3mz6', '5vi6'), ('Q9BY41', '5vi6', '3mz6'), ('Q9BZP6', '3rm4', '3rm9'), ('Q9BZP6', '3rm9', '3rm4'), ('Q9CPU0', '4kyh', '4kyk'), ('Q9CPU0', '4kyk', '4pv5'), ('Q9CPU0', '4pv5', '4kyk'), ('Q9E7M1', '3sm2', '3slz'), ('Q9F4L3', '3d7k', '3iae'), ('Q9F4L3', '3iae', '3d7k'), ('Q9FUZ2', '3m6r', '3pn4'), ('Q9FUZ2', '3pn4', '3m6r'), ('Q9FV53', '4je7', '4je8'), ('Q9FV53', '4je8', '4je7'), ('Q9GK12', '3ng4', '3nw3'), ('Q9GK12', '3nw3', '3ng4'), ('Q9GK12', '3o4k', '3ng4'), ('Q9GK12', '3usx', '3ng4'), ('Q9GK12', '4fnn', '3o4k'), ('Q9H2K2', '3kr8', '4j21'), ('Q9H2K2', '4iue', '4j22'), ('Q9H2K2', '4j21', '4kzq'), ('Q9H2K2', '4j22', '4j3l'), ('Q9H2K2', '4j3l', '3kr8'), ('Q9H2K2', '4kzq', '4j21'), ('Q9H2K2', '4kzu', '4j22'), ('Q9H8M2', '4xy8', '5ji8'), ('Q9H8M2', '5eu1', '5i7y'), ('Q9H8M2', '5f1h', '5ji8'), ('Q9H8M2', '5f25', '5igm'), ('Q9H8M2', '5f2p', '5f25'), ('Q9H8M2', '5i7x', '5f25'), ('Q9H8M2', '5i7y', '5f25'), ('Q9H8M2', '5igm', '5i7y'), ('Q9H8M2', '5ji8', '5f25'), ('Q9H9B1', '5tuz', '5vsf'), ('Q9H9B1', '5vsf', '5tuz'), ('Q9HGR1', '4ymg', '4ymh'), ('Q9HGR1', '4ymh', '4ymg'), ('Q9HPW4', '2cc7', '2ccc'), ('Q9HPW4', '2ccb', '2cc7'), ('Q9HPW4', '2ccc', '2ccb'), ('Q9HYN5', '2boj', '2jdm'), ('Q9HYN5', '2jdm', '2boj'), ('Q9HYN5', '2jdp', '2boj'), ('Q9HYN5', '2jdu', '2jdm'), ('Q9HYN5', '3zdv', '2jdm'), ('Q9JLU4', '5ovc', '5ovp'), ('Q9JLU4', '5ovp', '6exj'), ('Q9JLU4', '6exj', '5ovp'), ('Q9K169', '4uc5', '4uma'), ('Q9K169', '4uma', '4umb'), ('Q9K169', '4umb', '4uma'), ('Q9K169', '4umc', '4uc5'), ('Q9KS12', '3eeb', '3fzy'), ('Q9KS12', '3fzy', '3eeb'), ('Q9KU37', '2oxn', '3gs6'), ('Q9KU37', '3gs6', '2oxn'), ('Q9KU37', '3gsm', '2oxn'), ('Q9KWT6', '1y3n', '1y3p'), ('Q9KWT6', '1y3p', '1y3n'), ('Q9L5C8', '3g2y', '3g31'), ('Q9L5C8', '3g2z', '3g30'), ('Q9L5C8', '3g30', '3g31'), ('Q9L5C8', '3g31', '3g34'), ('Q9L5C8', '3g32', '3g35'), ('Q9L5C8', '3g34', '4de0'), ('Q9L5C8', '3g35', '3g32'), ('Q9L5C8', '4de0', '3g34'), ('Q9L5C8', '4de1', '3g30'), ('Q9L5C8', '4de2', '3g34'), ('Q9N1E2', '1g98', '1koj'), ('Q9N1E2', '1koj', '1g98'), ('Q9NPB1', '1q91', '6g2l'), ('Q9NPB1', '6g2l', '6g2m'), ('Q9NPB1', '6g2m', '1q91'), ('Q9NQG6', '4nxu', '4nxv'), ('Q9NQG6', '4nxv', '4nxu'), ('Q9NR97', '4r0a', '5wyz'), ('Q9NR97', '5wyx', '4r0a'), ('Q9NR97', '5wyz', '5wyx'), ('Q9NXS2', '3pb7', '3pb9'), ('Q9NXS2', '3pb8', '3pb9'), ('Q9NXS2', '3pb9', '3pb8'), ('Q9NY33', '3t6b', '5e3a'), ('Q9NY33', '5e3a', '3t6b'), ('Q9NZD2', '2euk', '2evl'), ('Q9NZD2', '2evl', '2euk'), ('Q9PTT3', '6gnm', '6gnp'), ('Q9PTT3', '6gnp', '6gnw'), ('Q9PTT3', '6gnr', '6gnw'), ('Q9PTT3', '6gnw', '6gon'), ('Q9PTT3', '6gon', '6gnm'), ('Q9QB59', '3lzs', '3lzu'), ('Q9QB59', '3lzu', '3lzs'), ('Q9QLL6', '5efa', '5efc'), ('Q9QLL6', '5efc', '5efa'), ('Q9QYJ6', '2o8h', '2ovv'), ('Q9QYJ6', '2ovv', '2ovy'), ('Q9QYJ6', '2ovy', '2o8h'), ('Q9R0G6', '3v2n', '3v2q'), ('Q9R0G6', '3v2p', '3v2n'), ('Q9R0G6', '3v2q', '3v2p'), ('Q9R4E4', '2pqb', '2pqc'), ('Q9R4E4', '2pqc', '2pqb'), ('Q9RA63', '4lj5', '4lj8'), ('Q9RA63', '4lj8', '4lj5'), ('Q9RS96', '5hva', '5hwu'), ('Q9RS96', '5hwu', '5hva'), ('Q9T0I8', '2qtg', '3lgs'), ('Q9T0I8', '2qtt', '3lgs'), ('Q9T0I8', '3lgs', '2qtt'), ('Q9U9J6', '3cyz', '3d78'), ('Q9U9J6', '3cz1', '3cyz'), ('Q9U9J6', '3d78', '3cz1'), ('Q9UBN7', '5kh3', '6ce6'), ('Q9UBN7', '6ce6', '5kh3'), ('Q9UBN7', '6ced', '5kh3'), ('Q9UGN5', '3kjd', '4zzx'), ('Q9UGN5', '4zzx', '4zzy'), ('Q9UGN5', '4zzy', '4zzx'), ('Q9UIF8', '4nra', '5e73'), ('Q9UIF8', '4rvr', '5mge'), ('Q9UIF8', '5e73', '5e74'), ('Q9UIF8', '5e74', '5mgf'), ('Q9UIF8', '5mge', '5e73'), ('Q9UIF9', '5mgj', '6fgg'), ('Q9UIF9', '5mgk', '5mgj'), ('Q9UM73', '2xb7', '5fto'), ('Q9UM73', '4cd0', '5aa9'), ('Q9UM73', '4clj', '4cmo'), ('Q9UM73', '4cmo', '2xb7'), ('Q9UM73', '5aa9', '5kz0'), ('Q9UM73', '5fto', '2xb7'), ('Q9UM73', '5kz0', '5aa9'), ('Q9VHA0', '2r58', '2r5a'), ('Q9VHA0', '2r5a', '2r58'), ('Q9VWX8', '5aan', '5fyx'), ('Q9VWX8', '5fyx', '6epa'), ('Q9VWX8', '6epa', '5aan'), ('Q9WUL6', '5t8o', '5t8p'), ('Q9WUL6', '5t8p', '5t8o'), ('Q9WYE2', '2zwz', '2zx8'), ('Q9WYE2', '2zx6', '2zwz'), ('Q9WYE2', '2zx7', '2zwz'), ('Q9WYE2', '2zx8', '2zx6'), ('Q9XEI3', '1x38', '1x39'), ('Q9XEI3', '1x39', '1x38'), ('Q9Y233', '3ui7', '4lm0'), ('Q9Y233', '3uuo', '4lkq'), ('Q9Y233', '4dff', '3ui7'), ('Q9Y233', '4hf4', '3uuo'), ('Q9Y233', '4lkq', '4lm0'), ('Q9Y233', '4llj', '4dff'), ('Q9Y233', '4llk', '3ui7'), ('Q9Y233', '4llp', '4llj'), ('Q9Y233', '4llx', '4llk'), ('Q9Y233', '4lm0', '4lkq'), ('Q9Y3Q0', '3fed', '3fee'), ('Q9Y3Q0', '3fee', '3ff3'), ('Q9Y3Q0', '3ff3', '3fee'), ('Q9Y5Y6', '2gv6', '2gv7'), ('Q9Y5Y6', '2gv7', '4jyt'), ('Q9Y5Y6', '4jyt', '4o9v'), ('Q9Y5Y6', '4jzi', '2gv7'), ('Q9Y5Y6', '4o97', '4jz1'), ('Q9Y657', '4h75', '5jsj'), ('Q9Y657', '5jsg', '5jsj'), ('Q9Y657', '5jsj', '5jsg'), ('Q9Y6F1', '3c4h', '3fhb'), ('Q9Y6F1', '3fhb', '3c4h'), ('Q9YFY3', '4rr6', '4rra'), ('Q9YFY3', '4rra', '4rr6'), ('Q9Z2X8', '5fnr', '5fnt'), ('Q9Z2X8', '5fns', '5fnt'), ('Q9Z2X8', '5fnt', '5fnu'), ('Q9Z2X8', '5fnu', '5fnr'), ('Q9Z4P9', '4d4d', '5n0f'), ('Q9Z4P9', '5m77', '4d4d'), ('Q9Z4P9', '5n0f', '4d4d'), ('Q9ZMY2', '4ffs', '4wkn'), ('Q9ZMY2', '4wkn', '4ynb'), ('Q9ZMY2', '4wko', '4ynb'), ('Q9ZMY2', '4wkp', '4ynb'), ('Q9ZMY2', '4ynb', '4wkn'), ('Q9ZMY2', '4yo8', '4wkn'), ('U5XBU0', '4cpy', '4cpz'), ('U5XBU0', '4cpz', '4cpy'), ('U6NCW5', '4ovf', '4ovh'), ('U6NCW5', '4ovg', '4ovh'), ('U6NCW5', '4ovh', '4ovg'), ('U6NCW5', '4pnu', '4ovf'), ('V5Y949', '4q1w', '4q1y'), ('V5Y949', '4q1x', '4q1y'), ('V5Y949', '4q1y', '4q1w'), ('V5YAB1', '5kr1', '5kr2'), ('V5YAB1', '5kr2', '5kr1'), ('W5R8B8', '5nze', '5nzn'), ('W5R8B8', '5nzf', '5nzn'), ('W5R8B8', '5nzn', '5nze')]
        grouped_files = group_files(args.n, process)
        run_all(args.docked_prot_file, args.run_path, args.raw_root, args.data_root, grouped_files, args.n)

    if args.task == 'group':
        # process = get_prots(args.docked_prot_file)
        process = [('P00883', '1ado', '2ot1'), ('P00883', '2ot1', '1ado'), ('P00915', '1azm', '3lxe'), ('P00915', '2nmx', '6f3b'), ('P00915', '2nn1', '3lxe'), ('P00915', '2nn7', '6evr'), ('P00915', '3lxe', '6faf'), ('P00915', '6evr', '6g3v'), ('P00915', '6ex1', '6evr'), ('P00915', '6f3b', '2nmx'), ('P00915', '6faf', '2nn7'), ('P00915', '6g3v', '3lxe'), ('P00918', '1avn', '1bnu'), ('P00918', '1bcd', '1bnt'), ('P00918', '1bn1', '1bnn'), ('P00918', '1bn3', '1bnq'), ('P00918', '1bn4', '1bnt'), ('P00918', '1bnn', '1bnv'), ('P00918', '1bnq', '1bnu'), ('P00918', '1bnt', '1bn1'), ('P00918', '1bnu', '1bnn'), ('P00918', '1bnv', '1bnq'), ('P00929', '1tjp', '2clh'), ('P00929', '2cle', '2clh'), ('P00929', '2clh', '2cli'), ('P00929', '2cli', '2clh'), ('P00929', '2clk', '1tjp'), ('P01011', '5om2', '6ftp'), ('P01011', '5om3', '5om2'), ('P01011', '5om7', '5om2'), ('P01011', '6ftp', '5om2'), ('P01112', '4ury', '6d5e'), ('P01112', '4urz', '4ury'), ('P01112', '6d55', '6d5g'), ('P01112', '6d56', '4ury'), ('P01112', '6d5e', '6d5g'), ('P01112', '6d5g', '6d55'), ('P01112', '6d5h', '6d56'), ('P01112', '6d5j', '6d5h'), ('P01116', '4dst', '4epy'), ('P01116', '4dsu', '4dst'), ('P01116', '4epy', '4dst'), ('P01116', '6fa4', '4epy'), ('P01724', '1dl7', '1oar'), ('P01724', '1oar', '1dl7'), ('P01834', '1a4k', '1i7z'), ('P01857', '1aj7', '1gaf'), ('P01857', '1gaf', '1aj7'), ('P01892', '3qfd', '5jzi'), ('P01892', '5isz', '5jzi'), ('P01892', '5jzi', '3qfd'), ('P01901', '1fo0', '1g7q'), ('P01901', '1fzj', '3p9m'), ('P01901', '1fzk', '1fzo'), ('P01901', '1fzm', '1fzj'), ('P01901', '1g7q', '3p9l'), ('P01901', '3p9l', '1fo0'), ('P01901', '3p9m', '1fzo'), ('P01901', '4pg9', '1fzk'), ('P02701', '2a5b', '2jgs'), ('P02701', '2a5c', '2a5b'), ('P02701', '2a8g', '2a5c'), ('P02701', '2jgs', '2a5b'), ('P02743', '2w08', '3kqr'), ('P02743', '3kqr', '4ayu'), ('P02743', '4avs', '3kqr'), ('P02743', '4ayu', '3kqr'), ('P02754', '1gx8', '3uew'), ('P02754', '3nq3', '3uew'), ('P02754', '3nq9', '3uew'), ('P02754', '3ueu', '3uex'), ('P02754', '3uev', '3nq3'), ('P02754', '3uew', '3ueu'), ('P02754', '3uex', '3nq9'), ('P02754', '4gny', '2gj5'), ('P02754', '6ge7', '3ueu'), ('P02766', '1bm7', '2f7i'), ('P02766', '1e4h', '2b9a'), ('P02766', '2b9a', '2f7i'), ('P02766', '2f7i', '1bm7'), ('P02766', '2g5u', '2f7i'), ('P02766', '3cfn', '1bm7'), ('P02766', '3cft', '1bm7'), ('P02766', '3kgt', '3kgu'), ('P02766', '3kgu', '1e4h'), ('P02766', '3nee', '1e4h'), ('P02768', '1hk4', '6ezq'), ('P02768', '6ezq', '1hk4'), ('P02791', '3f33', '3u90'), ('P02829', '1amw', '2weq'), ('P02829', '1bgq', '2weq'), ('P02829', '2cgf', '2vwc'), ('P02829', '2fxs', '2weq'), ('P02829', '2iwx', '2cgf'), ('P02829', '2vw5', '1amw'), ('P02829', '2vwc', '2iwx'), ('P02829', '2weq', '2yge'), ('P02829', '2wer', '2yge'), ('P02829', '2yge', '1bgq'), ('P02911', '1laf', '1lag'), ('P02911', '1lag', '1lah'), ('P02911', '1lah', '1lst'), ('P02911', '1lst', '1lag'), ('P02925', '1drj', '1drk'), ('P02925', '1drk', '1drj'), ('P02925', '2dri', '1drk'), ('P03176', '1e2k', '1e2l'), ('P03176', '1e2l', '1e2k'), ('P03366', '1ajv', '1ec1'), ('P03366', '1ajx', '1d4i'), ('P03366', '1d4h', '1ec1'), ('P03366', '1d4i', '1d4j'), ('P03366', '1d4j', '1d4h'), ('P03366', '1ebw', '1ec0'), ('P03366', '1eby', '1ec1'), ('P03366', '1ebz', '1eby'), ('P03366', '1ec0', '1d4h'), ('P03366', '1ec1', '1ebw'), ('P03367', '1a94', '1dif'), ('P03367', '1d4y', '1iiq'), ('P03367', '1hpo', '1mrw'), ('P03367', '1hpx', '1a94'), ('P03367', '1iiq', '1lzq'), ('P03367', '1lzq', '1a94'), ('P03367', '1m0b', '1hpo'), ('P03367', '1mrw', '1dif'), ('P03368', '1a9m', '1gnm'), ('P03368', '1gnm', '1g35'), ('P03368', '1gnn', '1zp8'), ('P03368', '1gno', '1zp8'), ('P03368', '1zp8', '1gnm'), ('P03368', '1zpa', '2hs1'), ('P03368', '2hs1', '1zpa'), ('P03368', '2i4d', '1gnm'), ('P03369', '1aid', '1z1h'), ('P03369', '1b6j', '1z1h'), ('P03369', '1b6k', '1d4l'), ('P03369', '1b6l', '1aid'), ('P03369', '1d4k', '1kzk'), ('P03369', '1d4l', '1d4k'), ('P03369', '1kzk', '3aid'), ('P03369', '1mtr', '1kzk'), ('P03369', '1z1h', '1d4k'), ('P03369', '3aid', '1aid'), ('P03372', '1qkt', '2qe4'), ('P03372', '2p15', '2qe4'), ('P03372', '2pog', '2qe4'), ('P03372', '4mgd', '2pog'), ('P03472', '1f8b', '2qwd'), ('P03472', '1f8c', '2qwe'), ('P03472', '1f8d', '2qwf'), ('P03472', '1f8e', '1f8d'), ('P03472', '2qwb', '1f8c'), ('P03472', '2qwc', '2qwf'), ('P03472', '2qwd', '2qwf'), ('P03472', '2qwe', '2qwd'), ('P03472', '2qwf', '1f8e'), ('P03951', '4cr5', '4cr9'), ('P03951', '4cr9', '4x6m'), ('P03951', '4cra', '4ty6'), ('P03951', '4crb', '4ty7'), ('P03951', '4crc', '4ty6'), ('P03951', '4crf', '4cr9'), ('P03951', '4ty6', '4cr9'), ('P03951', '4ty7', '4cra'), ('P03951', '4x6m', '4crb'), ('P03951', '4x6n', '4crc'), ('P03956', '1cgl', '966c'), ('P03956', '966c', '1cgl'), ('P03958', '1add', '1fkw'), ('P03958', '1fkw', '1add'), ('P04035', '3cct', '3cd7'), ('P04035', '3ccw', '3cda'), ('P04035', '3ccz', '3cd7'), ('P04035', '3cd0', '3cct'), ('P04035', '3cd5', '3cda'), ('P04035', '3cd7', '3cd0'), ('P04035', '3cda', '3cct'), ('P04035', '3cdb', '3cda'), ('P04058', '1e66', '1h23'), ('P04058', '1gpk', '1h23'), ('P04058', '1gpn', '5nau'), ('P04058', '1h22', '1gpk'), ('P04058', '1h23', '1gpk'), ('P04058', '3zv7', '1gpk'), ('P04058', '5bwc', '5nap'), ('P04058', '5nap', '1gpn'), ('P04058', '5nau', '3zv7'), ('P04117', '1adl', '2qm9'), ('P04117', '1g74', '2ans'), ('P04117', '2ans', '2qm9'), ('P04117', '2qm9', '1adl'), ('P04117', '3hk1', '1adl'), ('P04150', '4p6w', '4p6x'), ('P04150', '4p6x', '4p6w'), ('P04278', '1kdk', '1lhu'), ('P04278', '1lhu', '1kdk'), ('P04584', '1hii', '6upj'), ('P04584', '1hsh', '1ivp'), ('P04584', '1ivp', '1hii'), ('P04584', '5upj', '1ivp'), ('P04584', '6upj', '1hii'), ('P04585', '1a30', '1hwr'), ('P04585', '1bv9', '1bwb'), ('P04585', '1bwa', '1bwb'), ('P04585', '1dmp', '1bwb'), ('P04585', '1hvh', '1bwb'), ('P04585', '1hwr', '1dmp'), ('P04585', '1hxb', '1bwa'), ('P04587', '1bdq', '2aog'), ('P04587', '1g2k', '1hpv'), ('P04587', '1hpv', '2aoe'), ('P04587', '1tcx', '1hvs'), ('P04587', '2aoc', '1tcx'), ('P04587', '2aod', '1hpv'), ('P04587', '2aoe', '1hvj'), ('P04587', '2aog', '1bdq'), ('P04637', '2vuk', '5aba'), ('P04637', '4agl', '4agm'), ('P04637', '4agm', '4agq'), ('P04637', '4agn', '4agl'), ('P04637', '4ago', '4agl'), ('P04637', '4agq', '4agp'), ('P04637', '5aba', '4agl'), ('P04637', '5aoi', '4ago'), ('P04642', '4aj4', '4ajl'), ('P04642', '4aje', '4aj4'), ('P04642', '4aji', '4aje'), ('P04642', '4ajl', '4aji'), ('P04746', '1u33', '1xd0'), ('P04746', '1xd0', '3old'), ('P04746', '3old', '1xd0'), ('P04746', '4gqq', '3old'), ('P04746', '4gqr', '4gqq'), ('P04789', '1iih', '1trd'), ('P04789', '1kv5', '4tim'), ('P04789', '1trd', '1kv5'), ('P04789', '2j27', '1kv5'), ('P04789', '2v2c', '1iih'), ('P04789', '2v2h', '2v2c'), ('P04789', '4tim', '1iih'), ('P04816', '1usi', '1usk'), ('P04816', '1usk', '1usi'), ('P04905', '2gst', '3gst'), ('P04905', '3gst', '2gst'), ('P04995', '3hl8', '3hp9'), ('P04995', '3hp9', '3hl8'), ('P05089', '3f80', '3mfv'), ('P05089', '3kv2', '3lp7'), ('P05089', '3lp4', '3mjl'), ('P05089', '3lp7', '3mfw'), ('P05089', '3mfv', '3lp7'), ('P05089', '3mfw', '3kv2'), ('P05089', '3mjl', '3mfv'), ('P05091', '4kwf', '4kwg'), ('P05091', '4kwg', '4kwf'), ('P05413', '1hmr', '4tkb'), ('P05413', '1hms', '4tjz'), ('P05413', '1hmt', '1hmr'), ('P05413', '3wvm', '1hms'), ('P05413', '4tjz', '3wvm'), ('P05413', '4tkb', '1hmt'), ('P05413', '4tkh', '5hz9'), ('P05413', '4tkj', '4tjz'), ('P05413', '5hz9', '1hms'), ('P05543', '2xn3', '2xn5'), ('P05543', '2xn5', '2xn3'), ('P05981', '1o5e', '1p57'), ('P05981', '1p57', '1o5e'), ('P06202', '1b05', '1b40'), ('P06202', '1b0h', '1b3f'), ('P06202', '1b1h', '1b3h'), ('P06202', '1b2h', '1b3g'), ('P06202', '1b32', '1b3l'), ('P06202', '1b3f', '1b3h'), ('P06202', '1b3g', '1b3l'), ('P06202', '1b3h', '1b2h'), ('P06202', '1b3l', '1b3g'), ('P06202', '1b40', '1b2h'), ('P06239', '1bhf', '1lkk'), ('P06239', '1lkk', '1lkl'), ('P06239', '1lkl', '1lkk'), ('P06276', '6ep4', '6eqp'), ('P06276', '6eqp', '6ep4'), ('P06401', '1a28', '1sr7'), ('P06730', '2v8w', '5ei3'), ('P06730', '4tpw', '2v8w'), ('P06730', '5ei3', '4tpw'), ('P06875', '1ai4', '1ai5'), ('P06875', '1ai5', '1ajq'), ('P06875', '1ai7', '1ai4'), ('P06875', '1ajn', '1ai5'), ('P06875', '1ajp', '1ai4'), ('P06875', '1ajq', '1ai7'), ('P07267', '1fq5', '2jxr'), ('P07267', '2jxr', '1fq5'), ('P07445', '1e3v', '1ogx'), ('P07445', '1ogx', '5g2g'), ('P07445', '5g2g', '1ogx'), ('P07711', '3h89', '3h8b'), ('P07711', '3h8b', '3h89'), ('P07900', '1yc1', '2xab'), ('P07900', '1yc4', '2xdk'), ('P07900', '1yet', '2uwd'), ('P07900', '2qg0', '1yc1'), ('P07900', '2qg2', '1yet'), ('P07900', '2uwd', '1yc4'), ('P07900', '2xab', '1yc1'), ('P07900', '2xdk', '2xdx'), ('P07900', '2xdl', '1yc4'), ('P07900', '2xdx', '2xdk'), ('P07986', '1fh7', '1fh8'), ('P07986', '1fh8', '1fhd'), ('P07986', '1fh9', '1j01'), ('P07986', '1fhd', '1fh9'), ('P07986', '1j01', '1fhd'), ('P08191', '1uwf', '4x5q'), ('P08191', '4css', '4x50'), ('P08191', '4cst', '4x5r'), ('P08191', '4lov', '4cst'), ('P08191', '4x50', '4x5p'), ('P08191', '4x5p', '4css'), ('P08191', '4x5q', '4css'), ('P08191', '4x5r', '4css'), ('P08191', '4xo8', '4lov'), ('P08235', '2oax', '5mwy'), ('P08235', '5l7e', '2oax'), ('P08235', '5l7g', '5l7h'), ('P08235', '5mwp', '5l7g'), ('P08235', '5mwy', '5mwp'), ('P08238', '5uc4', '5ucj'), ('P08238', '5ucj', '5uc4'), ('P08254', '1b8y', '2usn'), ('P08254', '1ciz', '1hfs'), ('P08254', '1hfs', '1ciz'), ('P08254', '1sln', '2usn'), ('P08254', '1usn', '2usn'), ('P08254', '2d1o', '2usn'), ('P08254', '2usn', '1sln'), ('P08263', '1ydk', '4hj2'), ('P08263', '4hj2', '1ydk'), ('P08473', '1r1h', '1r1j'), ('P08473', '1r1j', '1r1h'), ('P08559', '3exe', '3exh'), ('P08559', '3exh', '3exe'), ('P08581', '2wgj', '3u6i'), ('P08581', '3q6w', '3zxz'), ('P08581', '3r7o', '3q6w'), ('P08581', '3u6h', '3u6i'), ('P08581', '3u6i', '3r7o'), ('P08581', '3zbx', '3u6i'), ('P08581', '3zc5', '3zcl'), ('P08581', '3zcl', '2wgj'), ('P08581', '3zxz', '3zc5'), ('P08709', '2b7d', '2bz6'), ('P08709', '2bz6', '5u6j'), ('P08709', '2flr', '4x8u'), ('P08709', '4ish', '4na9'), ('P08709', '4isi', '4x8u'), ('P08709', '4na9', '2flr'), ('P08709', '4x8u', '4ish'), ('P08709', '4x8v', '4ish'), ('P08709', '5l30', '4ish'), ('P08709', '5u6j', '4x8u'), ('P09211', '10gs', '3gss'), ('P09211', '1lbk', '5j41'), ('P09211', '2gss', '3gss'), ('P09211', '3gss', '5j41'), ('P09211', '5j41', '2gss'), ('P09237', '1mmq', '1mmr'), ('P09237', '1mmr', '1mmq'), ('P09382', '3oy8', '3oyw'), ('P09382', '3oyw', '3oy8'), ('P09455', '5ha1', '5hbs'), ('P09455', '5hbs', '5ha1'), ('P09464', '1lke', '1n0s'), ('P09464', '1lnm', '1n0s'), ('P09464', '1n0s', '1lnm'), ('P09874', '2rd6', '3gjw'), ('P09874', '3gjw', '5xsr'), ('P09874', '3l3l', '2rd6'), ('P09874', '3l3m', '5xsr'), ('P09874', '4und', '5xsr'), ('P09874', '4zzz', '2rd6'), ('P09874', '5xsr', '4und'), ('P09874', '6bhv', '4und'), ('P09958', '4omc', '4ryd'), ('P09958', '4ryd', '6eqx'), ('P09958', '6eqv', '6eqw'), ('P09958', '6eqw', '6eqv'), ('P09958', '6eqx', '4omc'), ('P09960', '2r59', '4l2l'), ('P09960', '3b7r', '2r59'), ('P09960', '3b7u', '3b7r'), ('P09960', '3fh7', '3b7r'), ('P09960', '4l2l', '3fh7'), ('P0A4Z6', '2xb8', '3n76'), ('P0A4Z6', '3n76', '3n7a'), ('P0A4Z6', '3n7a', '4ciw'), ('P0A4Z6', '3n86', '4b6o'), ('P0A4Z6', '3n8k', '3n76'), ('P0A4Z6', '4b6o', '2xb8'), ('P0A4Z6', '4b6p', '2xb8'), ('P0A4Z6', '4ciw', '3n76'), ('P0A4Z6', '4kiu', '3n86'), ('P0A538', '1g2o', '1n3i'), ('P0A538', '1n3i', '1g2o'), ('P0A546', '4km0', '4km2'), ('P0A546', '4km2', '4km0'), ('P0A5R0', '3cow', '3iod'), ('P0A5R0', '3coy', '3coz'), ('P0A5R0', '3coz', '3ime'), ('P0A5R0', '3imc', '3ioc'), ('P0A5R0', '3ime', '3ioc'), ('P0A5R0', '3iob', '3iod'), ('P0A5R0', '3ioc', '3iod'), ('P0A5R0', '3iod', '3coz'), ('P0A5R0', '3ioe', '3isj'), ('P0A5R0', '3isj', '3ioc'), ('P0A6D3', '1x8r', '1x8t'), ('P0A6D3', '1x8t', '1x8r'), ('P0A6D3', '2pq9', '1x8t'), ('P0A6I6', '6b7a', '6ckw'), ('P0A6I6', '6b7b', '6chp'), ('P0A6I6', '6chp', '6ckw'), ('P0A6I6', '6ckw', '6chp'), ('P0A6Y8', '4ezz', '4ezr'), ('P0A715', '1g7v', '1phw'), ('P0A715', '1phw', '1g7v'), ('P0A720', '4tmk', '5tmp'), ('P0A720', '5tmp', '4tmk'), ('P0A731', '1egh', '1ik4'), ('P0A731', '1ik4', '1s89'), ('P0A731', '1s89', '1egh'), ('P0A786', '1d09', '2fzk'), ('P0A786', '2fzc', '2fzk'), ('P0A786', '2fzk', '2h3e'), ('P0A786', '2h3e', '2fzc'), ('P0A884', '1f4e', '1f4f'), ('P0A884', '1f4f', '1f4e'), ('P0A884', '1f4g', '1f4f'), ('P0A8M3', '4hwo', '4hws'), ('P0A8M3', '4hwp', '4hws'), ('P0A8M3', '4hws', '4hwo'), ('P0A953', '2vb8', '2vba'), ('P0A953', '2vba', '2vb8'), ('P0A988', '4mjp', '4n9a'), ('P0A988', '4n9a', '4mjp'), ('P0ABF6', '1ctt', '1ctu'), ('P0ABF6', '1ctu', '1ctt'), ('P0ABP8', '1a69', '1k9s'), ('P0ABP8', '1k9s', '1a69'), ('P0ABQ4', '1dhi', '1dhj'), ('P0ABQ4', '1dhj', '1dhi'), ('P0ABQ4', '2drc', '1dhj'), ('P0AC14', '5u0w', '5u12'), ('P0AC14', '5u0y', '5u11'), ('P0AC14', '5u0z', '5v79'), ('P0AC14', '5u11', '5u0w'), ('P0AC14', '5u12', '5v7a'), ('P0AC14', '5u13', '5u0w'), ('P0AC14', '5u14', '5u11'), ('P0AC14', '5v79', '5u0y'), ('P0AC14', '5v7a', '5u0z'), ('P0AFG9', '3lpl', '3lq2'), ('P0AFG9', '3lq2', '3lpl'), ('P0C6F2', '3th9', '4mc1'), ('P0C6F2', '3vfa', '4mc9'), ('P0C6F2', '4i8w', '4qgi'), ('P0C6F2', '4i8z', '4mc2'), ('P0C6F2', '4mc1', '4mc9'), ('P0C6F2', '4mc2', '3th9'), ('P0C6F2', '4mc6', '4mc9'), ('P0C6F2', '4mc9', '3th9'), ('P0C6F2', '4qgi', '4i8w'), ('P0DMV8', '5aqz', '5mks'), ('P0DMV8', '5mkr', '5aqz'), ('P0DMV8', '5mks', '6fhk'), ('P0DMV8', '6fhk', '5aqz'), ('P0DOX7', '6msy', '6mub'), ('P0DOX7', '6mu3', '6msy'), ('P0DOX7', '6mub', '6msy'), ('P10153', '1hi3', '1hi5'), ('P10153', '1hi4', '1hi3'), ('P10153', '1hi5', '1hi4'), ('P10153', '5e13', '1hi4'), ('P10253', '5nn5', '5nn6'), ('P10253', '5nn6', '5nn5'), ('P10275', '1e3g', '3b5r'), ('P10275', '1xow', '3b66'), ('P10275', '1z95', '3b65'), ('P10275', '2ax9', '3b5r'), ('P10275', '3b5r', '1z95'), ('P10275', '3b66', '3b67'), ('P10275', '3b67', '1z95'), ('P10275', '3b68', '3b65'), ('P10275', '5cj6', '1xow'), ('P10415', '4ieh', '6gl8'), ('P10415', '4lvt', '6gl8'), ('P10415', '4lxd', '6gl8'), ('P10415', '6gl8', '4ieh'), ('P10845', '3c88', '3c89'), ('P10845', '3c89', '3c88'), ('P11021', '3ldp', '5f2r'), ('P11021', '5evz', '5ey4'), ('P11021', '5exw', '5f2r'), ('P11021', '5ey4', '3ldp'), ('P11021', '5f1x', '3ldp'), ('P11021', '5f2r', '3ldp'), ('P11142', '3ldq', '3m3z'), ('P11142', '3m3z', '3ldq'), ('P11309', '1xws', '4k0y'), ('P11309', '2c3i', '3bgq'), ('P11309', '2xj1', '2xj2'), ('P11309', '2xj2', '3bgz'), ('P11309', '3bgq', '3jya'), ('P11309', '3bgz', '2c3i'), ('P11309', '3jy0', '3bgz'), ('P11309', '3jya', '4k18'), ('P11309', '4k0y', '1xws'), ('P11309', '4k18', '2c3i'), ('P11362', '4rwj', '5am6'), ('P11362', '5am6', '4rwj'), ('P11362', '5am7', '5am6'), ('P11444', '3uxk', '3uxl'), ('P11444', '3uxl', '3uxk'), ('P11444', '4fp1', '4m6u'), ('P11444', '4m6u', '4fp1'), ('P11473', '1ie9', '1s19'), ('P11473', '1s19', '1ie9'), ('P11588', '1qy1', '1qy2'), ('P11588', '1qy2', '1qy1'), ('P11838', '1epo', '3wz7'), ('P11838', '1gvw', '3pww'), ('P11838', '1gvx', '3wz7'), ('P11838', '2v00', '1epo'), ('P11838', '3prs', '1epo'), ('P11838', '3pww', '2v00'), ('P11838', '3uri', '3wz8'), ('P11838', '3wz6', '1gvx'), ('P11838', '3wz7', '2v00'), ('P11838', '3wz8', '1gvw'), ('P12497', '4ahr', '4ahu'), ('P12497', '4ahs', '4ahr'), ('P12497', '4ahu', '4ahs'), ('P12499', '1hxw', '4ejl'), ('P12499', '1pro', '4ej8'), ('P12499', '4ej8', '1pro'), ('P12499', '4ejl', '1pro'), ('P12694', '1ols', '1olx'), ('P12694', '1olu', '1olx'), ('P12694', '1olx', '1v16'), ('P12694', '1v11', '1ols'), ('P12694', '1v16', '1olu'), ('P12694', '1v1m', '1v11'), ('P12821', '2oc2', '3bkl'), ('P12821', '2xyd', '6en5'), ('P12821', '3bkk', '6f9u'), ('P12821', '3bkl', '3nxq'), ('P12821', '3l3n', '4ca5'), ('P12821', '3nxq', '3bkl'), ('P12821', '4ca5', '3bkl'), ('P12821', '4ca6', '6f9u'), ('P12821', '6en5', '6f9u'), ('P12821', '6f9u', '3nxq'), ('P13009', '6bdy', '6bm6'), ('P13009', '6bm5', '6bdy'), ('P13009', '6bm6', '6bm5'), ('P13053', '1rjk', '2o4j'), ('P13053', '2o4j', '2o4r'), ('P13053', '2o4r', '2o4j'), ('P13482', '2jf4', '2jjb'), ('P13482', '2jg0', '2jjb'), ('P13482', '2jjb', '2jg0'), ('P13491', '4i8x', '4i9u'), ('P13491', '4i9h', '4i9u'), ('P13491', '4i9u', '4i8x'), ('P13631', '1fcx', '1fcz'), ('P13631', '1fcy', '1fcz'), ('P13631', '1fcz', '1fd0'), ('P13631', '1fd0', '1fcz'), ('P14061', '1i5r', '3hb4'), ('P14061', '3hb4', '1i5r'), ('P14174', '4wrb', '6cbf'), ('P14174', '5hvs', '6cbg'), ('P14174', '5hvt', '5hvs'), ('P14174', '5j7q', '4wrb'), ('P14174', '6b1k', '6cbf'), ('P14174', '6cbf', '6b1k'), ('P14174', '6cbg', '6b1k'), ('P14207', '4kmz', '4kn1'), ('P14207', '4kn0', '4kn1'), ('P14207', '4kn1', '4kn0'), ('P14324', '1yq7', '4pvy'), ('P14324', '2f94', '4pvy'), ('P14324', '2f9k', '1yq7'), ('P14324', '4pvx', '2f94'), ('P14324', '4pvy', '4pvx'), ('P14324', '5ja0', '2f94'), ('P14751', '1nw5', '1nw7'), ('P14751', '1nw7', '1nw5'), ('P14769', '1gwv', '1o7o'), ('P14769', '1o7o', '1gwv'), ('P15090', '2hnx', '5d45'), ('P15090', '5d45', '5y13'), ('P15090', '5d47', '5y13'), ('P15090', '5d48', '5d45'), ('P15090', '5edb', '2hnx'), ('P15090', '5edc', '2hnx'), ('P15090', '5hz6', '5y12'), ('P15090', '5hz8', '5edc'), ('P15090', '5y12', '5d47'), ('P15090', '5y13', '5y12'), ('P15207', '1i37', '3g0w'), ('P15207', '2ihq', '1i37'), ('P15207', '3g0w', '2ihq'), ('P15379', '4mre', '4mrg'), ('P15379', '4mrg', '4np2'), ('P15379', '4np2', '4mrg'), ('P15379', '4np3', '4np2'), ('P15917', '1yqy', '4dv8'), ('P15917', '4dv8', '1yqy'), ('P16088', '1fiv', '2hah'), ('P16088', '2hah', '1fiv'), ('P16404', '1ax0', '3n35'), ('P16404', '3n35', '1ax0'), ('P16932', '1m0n', '1zc9'), ('P16932', '1m0o', '1m0q'), ('P16932', '1m0q', '1zc9'), ('P16932', '1zc9', '1m0n'), ('P17050', '4do4', '4do5'), ('P17050', '4do5', '4do4'), ('P17612', '3agl', '5izj'), ('P17612', '4uj1', '4ujb'), ('P17612', '4uj2', '4uj1'), ('P17612', '4uja', '4uj2'), ('P17612', '4ujb', '3agl'), ('P17612', '5izf', '4uj2'), ('P17612', '5izj', '4uja'), ('P17752', '3hf8', '3hfb'), ('P17752', '3hfb', '3hf8'), ('P17931', '1kjr', '3t1m'), ('P17931', '3t1m', '6eol'), ('P17931', '5e89', '6eog'), ('P17931', '5h9r', '5e89'), ('P17931', '6eog', '1kjr'), ('P17931', '6eol', '5h9r'), ('P18031', '1bzc', '1ecv'), ('P18031', '1bzj', '1c84'), ('P18031', '1c83', '1bzc'), ('P18031', '1c84', '1bzc'), ('P18031', '1c86', '1bzc'), ('P18031', '1c87', '1ecv'), ('P18031', '1c88', '1bzj'), ('P18031', '1ecv', '1bzc'), ('P18031', '1g7f', '1ecv'), ('P18031', '1g7g', '1c83'), ('P18654', '3ubd', '4gue'), ('P18654', '4gue', '3ubd'), ('P18670', '1ugx', '1ws4'), ('P18670', '1ws4', '1ugx'), ('P19491', '1ftm', '1my4'), ('P19491', '1my4', '1p1o'), ('P19491', '1p1n', '1wvj'), ('P19491', '1p1o', '1syh'), ('P19491', '1p1q', '1p1o'), ('P19491', '1syh', '1syi'), ('P19491', '1syi', '1p1q'), ('P19491', '1wvj', '1p1o'), ('P19491', '1xhy', '1syi'), ('P19491', '2al5', '1syi'), ('P19492', '3dln', '3rt8'), ('P19492', '3dp4', '4f39'), ('P19492', '3rt8', '3dln'), ('P19492', '4f39', '3rt8'), ('P19493', '3fas', '3fat'), ('P19493', '3fat', '3fas'), ('P19793', '3ozj', '4m8e'), ('P19793', '4k4j', '4m8e'), ('P19793', '4k6i', '3ozj'), ('P19793', '4m8e', '4pp5'), ('P19793', '4m8h', '1fm9'), ('P19793', '4poj', '4poh'), ('P19793', '4pp3', '1fm9'), ('P19793', '4pp5', '4poh'), ('P19812', '3nik', '3nim'), ('P19812', '3nim', '3nik'), ('P19971', '1uou', '2wk6'), ('P19971', '2wk6', '1uou'), ('P20371', '1eoc', '2buv'), ('P20371', '2buv', '1eoc'), ('P20701', '1rd4', '3f78'), ('P20701', '3f78', '1rd4'), ('P20906', '3f6e', '3fzn'), ('P20906', '3fzn', '3f6e'), ('P21675', '5i1q', '5mg2'), ('P21675', '5i29', '5i1q'), ('P21675', '5mg2', '5i1q'), ('P21836', '1n5r', '2whp'), ('P21836', '1q84', '4ara'), ('P21836', '2ha2', '2ha3'), ('P21836', '2ha3', '1n5r'), ('P21836', '2ha6', '2ha3'), ('P21836', '2whp', '2ha3'), ('P21836', '4ara', '2ha6'), ('P21836', '4arb', '1q84'), ('P21836', '5ehq', '1n5r'), ('P22102', '1njs', '4ew2'), ('P22102', '4ew2', '4ew3'), ('P22102', '4ew3', '1njs'), ('P22303', '4m0e', '4m0f'), ('P22303', '4m0f', '4m0e'), ('P22392', '3bbb', '3bbf'), ('P22392', '3bbf', '3bbb'), ('P22498', '1uwt', '1uwu'), ('P22498', '1uwu', '1uwt'), ('P22498', '2ceq', '2cer'), ('P22498', '2cer', '2ceq'), ('P22629', '1df8', '1swr'), ('P22629', '1sld', '3rdo'), ('P22629', '1srg', '3wzn'), ('P22629', '1str', '1df8'), ('P22629', '1swg', '1swr'), ('P22629', '1swr', '1srg'), ('P22629', '2izl', '3wzn'), ('P22629', '3rdo', '1sld'), ('P22629', '3rdq', '3wzn'), ('P22629', '3wzn', '1srg'), ('P22734', '3hvi', '4p58'), ('P22734', '3hvj', '3u81'), ('P22734', '3oe4', '5k03'), ('P22734', '3oe5', '4p58'), ('P22734', '3ozr', '5k03'), ('P22734', '3ozs', '3oe4'), ('P22734', '3ozt', '3ozs'), ('P22734', '3u81', '5k03'), ('P22734', '4p58', '5k03'), ('P22734', '5k03', '3hvj'), ('P22756', '1vso', '4dld'), ('P22756', '2f34', '3gba'), ('P22756', '2f35', '3gbb'), ('P22756', '2pbw', '4dld'), ('P22756', '2wky', '3s2v'), ('P22756', '3gba', '2wky'), ('P22756', '3gbb', '2f35'), ('P22756', '3s2v', '2wky'), ('P22756', '4dld', '2f35'), ('P22756', '4e0x', '2pbw'), ('P22887', '1s5z', '4cp5'), ('P22887', '4cp5', '1s5z'), ('P22894', '1jao', '1zs0'), ('P22894', '1jaq', '3tt4'), ('P22894', '1zs0', '1jaq'), ('P22894', '1zvx', '1jao'), ('P22894', '3tt4', '1zvx'), ('P23458', '4e4l', '4ivb'), ('P23458', '4e4n', '4ivc'), ('P23458', '4e5w', '4ivd'), ('P23458', '4ehz', '4ei4'), ('P23458', '4ei4', '4ivd'), ('P23458', '4fk6', '4ehz'), ('P23458', '4i5c', '4ivc'), ('P23458', '4ivb', '4e4n'), ('P23458', '4ivc', '4i5c'), ('P23458', '4ivd', '4fk6'), ('P23616', '4bt3', '4bt5'), ('P23616', '4bt4', '4bt5'), ('P23616', '4bt5', '4bt3'), ('P23946', '1t31', '5yjm'), ('P23946', '3n7o', '5yjm'), ('P23946', '5yjm', '1t31'), ('P24182', '2v58', '2v59'), ('P24182', '2v59', '3rv4'), ('P24182', '3rv4', '2v59'), ('P24247', '1jys', '1nc3'), ('P24247', '1nc1', '1y6q'), ('P24247', '1nc3', '1jys'), ('P24247', '1y6q', '1y6r'), ('P24247', '1y6r', '1jys'), ('P24941', '1b38', '1e1x'), ('P24941', '1e1v', '1pxo'), ('P24941', '1e1x', '1h1s'), ('P24941', '1h1s', '1pxp'), ('P24941', '1jsv', '1pxp'), ('P24941', '1pxn', '1jsv'), ('P24941', '1pxo', '1pxn'), ('P24941', '1pxp', '1jsv'), ('P24941', '2exm', '1pxo'), ('P24941', '2fvd', '2exm'), ('P25440', '2ydw', '2yek'), ('P25440', '2yek', '2ydw'), ('P25440', '4mr6', '4uyf'), ('P25440', '4qev', '4uyf'), ('P25440', '4qew', '4uyf'), ('P25440', '4uyf', '4mr6'), ('P25774', '2f1g', '3n3g'), ('P25774', '2hhn', '3n3g'), ('P25774', '3n3g', '2f1g'), ('P26281', '1ex8', '4f7v'), ('P26281', '4f7v', '1ex8'), ('P26514', '1od8', '1v0k'), ('P26514', '1v0k', '1od8'), ('P26514', '1v0l', '1od8'), ('P26662', '3p8n', '3p8o'), ('P26662', '3p8o', '3p8n'), ('P26663', '1nhu', '3mf5'), ('P26663', '1os5', '1nhu'), ('P26663', '3cj2', '1nhu'), ('P26663', '3cj5', '3cj4'), ('P26918', '2gkl', '3iog'), ('P26918', '3iof', '3iog'), ('P26918', '3iog', '2gkl'), ('P27487', '1n1m', '2ole'), ('P27487', '2oag', '4lko'), ('P27487', '2ole', '4jh0'), ('P27487', '3sww', '4lko'), ('P27487', '4jh0', '2oag'), ('P27487', '4lko', '2oag'), ('P27487', '5kby', '3sww'), ('P27694', '4luz', '4o0a'), ('P27694', '4o0a', '4r4c'), ('P27694', '4r4c', '4r4t'), ('P27694', '4r4i', '4r4t'), ('P27694', '4r4o', '4r4i'), ('P27694', '4r4t', '4o0a'), ('P27694', '5e7n', '4r4o'), ('P28482', '2ojg', '3i5z'), ('P28482', '2ojj', '1pme'), ('P28482', '3i5z', '2ojg'), ('P28482', '3i60', '2ojg'), ('P28482', '4qp2', '2ojg'), ('P28523', '1m2p', '1zog'), ('P28523', '1m2q', '2oxd'), ('P28523', '1m2r', '1zog'), ('P28523', '1om1', '2oxy'), ('P28523', '1zoe', '1om1'), ('P28523', '1zog', '1m2p'), ('P28523', '1zoh', '2oxy'), ('P28523', '2oxd', '1om1'), ('P28523', '2oxx', '1om1'), ('P28523', '2oxy', '1om1'), ('P28720', '1enu', '1k4h'), ('P28720', '1f3e', '1k4h'), ('P28720', '1k4g', '2z1w'), ('P28720', '1k4h', '1f3e'), ('P28720', '1q65', '1s39'), ('P28720', '1r5y', '1s39'), ('P28720', '1s38', '1enu'), ('P28720', '1s39', '2z1w'), ('P28720', '2qzr', '1k4h'), ('P28720', '2z1w', '1q65'), ('P29029', '2uy3', '2uy5'), ('P29029', '2uy4', '2uy3'), ('P29029', '2uy5', '2uy4'), ('P29317', '5i9x', '5i9z'), ('P29317', '5i9y', '5ia1'), ('P29317', '5i9z', '5ia2'), ('P29317', '5ia0', '5i9y'), ('P29317', '5ia1', '5njz'), ('P29317', '5ia2', '5njz'), ('P29317', '5ia3', '5i9y'), ('P29317', '5ia4', '5i9x'), ('P29317', '5ia5', '5i9y'), ('P29317', '5njz', '5ia3'), ('P29375', '5ivc', '5ivv'), ('P29375', '5ive', '5ivv'), ('P29375', '5ivv', '5ivc'), ('P29375', '5ivy', '5ivv'), ('P29375', '6dq4', '5ive'), ('P29498', '1vyf', '1vyg'), ('P29498', '1vyg', '1vyf'), ('P29597', '3nyx', '4gj2'), ('P29597', '4gfo', '4gii'), ('P29597', '4gih', '4gj2'), ('P29597', '4gii', '4gfo'), ('P29597', '4gj2', '4gj3'), ('P29597', '4gj3', '4gfo'), ('P29597', '4wov', '3nyx'), ('P29597', '5wal', '4gii'), ('P29724', '2fqw', '2fqx'), ('P29724', '2fqx', '2fqw'), ('P29724', '2fqy', '2fqw'), ('P29736', '1e6q', '1e6s'), ('P29736', '1e6s', '1e6q'), ('P30044', '4k7i', '4k7o'), ('P30044', '4k7n', '4k7i'), ('P30044', '4k7o', '4k7i'), ('P30044', '4mmm', '4k7o'), ('P30113', '2c80', '2ca8'), ('P30113', '2ca8', '2c80'), ('P30291', '5vc3', '5vc4'), ('P30291', '5vc4', '5vd2'), ('P30291', '5vd2', '5vc3'), ('P30967', '3tk2', '4jpx'), ('P30967', '4jpx', '4jpy'), ('P30967', '4jpy', '4jpx'), ('P31151', '2wor', '2wos'), ('P31151', '2wos', '2wor'), ('P31947', '3iqu', '5btv'), ('P31947', '5btv', '3iqu'), ('P31992', '1gyx', '1gyy'), ('P31992', '1gyy', '1gyx'), ('P32890', '1jqy', '1pzi'), ('P32890', '1pzi', '1jqy'), ('P33038', '3upk', '3v4t'), ('P33038', '3v4t', '3upk'), ('P33981', '3hmp', '5n93'), ('P33981', '5mrb', '5n93'), ('P33981', '5n84', '5n93'), ('P33981', '5n93', '5n84'), ('P35030', '1h4w', '5tp0'), ('P35030', '5tp0', '1h4w'), ('P35031', '1utj', '1utl'), ('P35031', '1utl', '1utm'), ('P35031', '1utm', '1utj'), ('P35120', '4pow', '4pp0'), ('P35120', '4pox', '5ota'), ('P35120', '4pp0', '5ito'), ('P35120', '5ito', '5ota'), ('P35120', '5itp', '5ito'), ('P35120', '5ot8', '5ota'), ('P35120', '5ot9', '4pp0'), ('P35120', '5ota', '4pow'), ('P35120', '5otc', '5ot8'), ('P35439', '1pb8', '1y1z'), ('P35439', '1pb9', '1y1z'), ('P35439', '1pbq', '5vih'), ('P35439', '1y1z', '5vih'), ('P35439', '1y20', '5u8c'), ('P35439', '4kfq', '5u8c'), ('P35439', '5dex', '1pb8'), ('P35439', '5u8c', '4kfq'), ('P35439', '5vih', '1pbq'), ('P35439', '5vij', '5dex'), ('P35505', '1hyo', '2hzy'), ('P35505', '2hzy', '1hyo'), ('P35790', '4cg8', '4da5'), ('P35790', '4cg9', '4cga'), ('P35790', '4cga', '5eqe'), ('P35790', '4da5', '4cg9'), ('P35790', '5afv', '4da5'), ('P35790', '5eqe', '4cg9'), ('P35790', '5eqp', '4br3'), ('P35790', '5eqy', '3zm9'), ('P35963', '1k6c', '1t7j'), ('P35963', '1k6p', '1k6v'), ('P35963', '1k6t', '1k6c'), ('P35963', '1k6v', '1k6c'), ('P35963', '1t7j', '1k6p'), ('P35968', '2qu6', '3vhk'), ('P35968', '3vhk', '4asd'), ('P35968', '4ag8', '2qu6'), ('P35968', '4agc', '3vhk'), ('P35968', '4asd', '4agc'), ('P35968', '4ase', '3vhk'), ('P36186', '4bi6', '4bi7'), ('P36186', '4bi7', '4bi6'), ('P36639', '4c9x', '6f20'), ('P36639', '5ant', '6f20'), ('P36639', '5anu', '5fsn'), ('P36639', '5anv', '4c9x'), ('P36639', '5fsn', '5anu'), ('P36639', '5fso', '6f20'), ('P36639', '6f20', '5ant'), ('P37231', '2i4j', '4a4v'), ('P37231', '2i4z', '2yfe'), ('P37231', '2p4y', '4a4w'), ('P37231', '2yfe', '2p4y'), ('P37231', '3b1m', '2yfe'), ('P37231', '3fur', '2i4z'), ('P37231', '3u9q', '4r06'), ('P37231', '4a4v', '2yfe'), ('P37231', '4a4w', '2i4z'), ('P37231', '4r06', '4a4w'), ('P38998', '2qrk', '2qrl'), ('P38998', '2qrl', '2qrk'), ('P39086', '3fuz', '3fvk'), ('P39086', '3fv1', '3fuz'), ('P39086', '3fv2', '3fv1'), ('P39086', '3fvk', '3fvn'), ('P39086', '3fvn', '3fv1'), ('P39900', '1rmz', '3f1a'), ('P39900', '2hu6', '3ehx'), ('P39900', '3ehx', '2hu6'), ('P39900', '3ehy', '3f18'), ('P39900', '3f15', '3f19'), ('P39900', '3f16', '3f1a'), ('P39900', '3f17', '3f15'), ('P39900', '3f18', '1rmz'), ('P39900', '3f19', '3f18'), ('P39900', '3f1a', '2hu6'), ('P42260', '2xxr', '2xxt'), ('P42260', '2xxt', '2xxr'), ('P42260', '2xxx', '2xxr'), ('P42264', '3s9e', '4g8n'), ('P42264', '4g8n', '4nwc'), ('P42264', '4nwc', '4g8n'), ('P42336', '5dxt', '5uk8'), ('P42336', '5uk8', '5dxt'), ('P42530', '2vmc', '2vmd'), ('P42530', '2vmd', '2vmc'), ('P43166', '3mdz', '6h36'), ('P43166', '3ml5', '6h38'), ('P43166', '6h36', '6h38'), ('P43166', '6h37', '3mdz'), ('P43166', '6h38', '3ml5'), ('P43405', '3fqe', '4fl2'), ('P43405', '4fl1', '4fl2'), ('P43405', '4fl2', '3fqe'), ('P43490', '2gvj', '5upf'), ('P43490', '4n9c', '5upf'), ('P43490', '5upe', '4n9c'), ('P43490', '5upf', '2gvj'), ('P43619', '3c2f', '3c2r'), ('P43619', '3c2o', '3c2r'), ('P43619', '3c2r', '3c2o'), ('P44539', '1f73', '1f74'), ('P44539', '1f74', '1f73'), ('P44542', '2cex', '3b50'), ('P44542', '3b50', '2cex'), ('P45446', '1n4h', '1nq7'), ('P45446', '1nq7', '1n4h'), ('P45452', '2d1n', '3ljz'), ('P45452', '3kek', '2d1n'), ('P45452', '3ljz', '3tvc'), ('P45452', '3tvc', '2d1n'), ('P45452', '456c', '2d1n'), ('P45452', '4l19', '3tvc'), ('P46925', '1lee', '1lf2'), ('P46925', '1lf2', '1lee'), ('P47205', '2ves', '4lch'), ('P47205', '4lch', '2ves'), ('P47205', '5drr', '4lch'), ('P47228', '1kmy', '1lgt'), ('P47228', '1lgt', '1kmy'), ('P47811', '4loo', '2ewa'), ('P48499', '1amk', '2vxn'), ('P48499', '2vxn', '1amk'), ('P48825', '4iic', '4iif'), ('P48825', '4iid', '4iif'), ('P48825', '4iie', '4iif'), ('P48825', '4iif', '4iie'), ('P49336', '4crl', '4f6w'), ('P49336', '4f6u', '4crl'), ('P49336', '4f6w', '4crl'), ('P49610', '4az5', '4az6'), ('P49610', '4az6', '4azg'), ('P49610', '4azb', '4az5'), ('P49610', '4azc', '4azg'), ('P49610', '4azg', '4azi'), ('P49610', '4azi', '4azb'), ('P49773', '5i2e', '5wa9'), ('P49773', '5i2f', '5wa9'), ('P49773', '5ipc', '5wa8'), ('P49773', '5kly', '5ipc'), ('P49773', '5kma', '5wa8'), ('P49773', '5wa8', '5i2f'), ('P49773', '5wa9', '5kly'), ('P49841', '1q5k', '4acc'), ('P49841', '3i4b', '4acc'), ('P49841', '4acc', '1q5k'), ('P50053', '5wbm', '5wbo'), ('P50053', '5wbo', '5wbm'), ('P51449', '4ymq', '5ufr'), ('P51449', '5g45', '4ymq'), ('P51449', '5g46', '5vb5'), ('P51449', '5ufr', '5vb7'), ('P51449', '5vb5', '5g45'), ('P51449', '5vb6', '5ufr'), ('P51449', '5vb7', '5g46'), ('P51449', '6cn5', '5g45'), ('P52293', '4u54', '4u5s'), ('P52293', '4u5n', '4u5s'), ('P52293', '4u5o', '4u54'), ('P52293', '4u5s', '4u5o'), ('P52333', '3lxk', '5lwm'), ('P52333', '5lwm', '6gl9'), ('P52333', '6gl9', '6glb'), ('P52333', '6gla', '6gl9'), ('P52333', '6glb', '5lwm'), ('P52679', '4rpn', '4rpo'), ('P52679', '4rpo', '4rpn'), ('P52699', '2doo', '5ev8'), ('P52699', '5ev8', '5ewa'), ('P52699', '5ewa', '5ev8'), ('P52700', '2fu8', '5dpx'), ('P52700', '2qdt', '5evb'), ('P52700', '5dpx', '5evk'), ('P52700', '5evb', '5evk'), ('P52700', '5evd', '5evk'), ('P52700', '5evk', '5evd'), ('P52732', '2x2r', '5zo8'), ('P52732', '5zo8', '2x2r'), ('P53350', '3fvh', '4e67'), ('P53350', '4e67', '3fvh'), ('P53350', '4o6w', '4o9w'), ('P53350', '4o9w', '4o6w'), ('P53582', '4u1b', '4u69'), ('P53582', '4u69', '4u73'), ('P53582', '4u6c', '4u6z'), ('P53582', '4u6w', '4u69'), ('P53582', '4u6z', '4u70'), ('P53582', '4u70', '4u73'), ('P53582', '4u71', '4u6z'), ('P53582', '4u73', '4u1b'), ('P54760', '6fni', '6fnj'), ('P54760', '6fnj', '6fni'), ('P54818', '4ufh', '4ufi'), ('P54818', '4ufi', '4ufj'), ('P54818', '4ufj', '4ufh'), ('P54818', '4ufk', '4ufl'), ('P54818', '4ufl', '4ufm'), ('P54818', '4ufm', '4ufj'), ('P54829', '5ovr', '5ovx'), ('P54829', '5ovx', '5ovr'), ('P55055', '4rak', '5jy3'), ('P55055', '5jy3', '4rak'), ('P55072', '3hu3', '4ko8'), ('P55072', '4ko8', '3hu3'), ('P55201', '4uye', '5ov8'), ('P55201', '5eq1', '5mwh'), ('P55201', '5etb', '4uye'), ('P55201', '5mwh', '5ov8'), ('P55201', '5o5a', '5ov8'), ('P55201', '5ov8', '4uye'), ('P55201', '6ekq', '4uye'), ('P55212', '4n5d', '4nbk'), ('P55212', '4n6g', '4nbk'), ('P55212', '4n7m', '4n6g'), ('P55212', '4nbk', '4n6g'), ('P55212', '4nbl', '4n6g'), ('P55212', '4nbn', '4n6g'), ('P55859', '1a9q', '1lvu'), ('P55859', '1b8n', '1vfn'), ('P55859', '1b8o', '1lvu'), ('P55859', '1lvu', '1b8n'), ('P55859', '1v48', '1b8n'), ('P55859', '1vfn', '1a9q'), ('P55859', '3fuc', '1a9q'), ('P56109', '3c52', '3c56'), ('P56109', '3c56', '3c52'), ('P56221', '2std', '5std'), ('P56221', '3std', '2std'), ('P56221', '4std', '7std'), ('P56221', '5std', '3std'), ('P56221', '6std', '4std'), ('P56221', '7std', '4std'), ('P56658', '1ndv', '1o5r'), ('P56658', '1ndw', '1v7a'), ('P56658', '1ndy', '1o5r'), ('P56658', '1ndz', '1o5r'), ('P56658', '1o5r', '1ndw'), ('P56658', '1qxl', '1uml'), ('P56658', '1uml', '1v7a'), ('P56658', '1v7a', '1qxl'), ('P56658', '2e1w', '1ndv'), ('P56817', '1fkn', '2vkm'), ('P56817', '1m4h', '2qmg'), ('P56817', '2fdp', '2g94'), ('P56817', '2g94', '2fdp'), ('P56817', '2p4j', '3bug'), ('P56817', '2qmg', '2fdp'), ('P56817', '2vkm', '3buf'), ('P56817', '3bra', '2g94'), ('P56817', '3buf', '1m4h'), ('P56817', '3bug', '2p4j'), ('P58154', '1uv6', '3wtj'), ('P58154', '1uw6', '3wtm'), ('P58154', '3u8j', '3wtj'), ('P58154', '3u8k', '1uv6'), ('P58154', '3u8l', '3wtl'), ('P58154', '3u8n', '3wtj'), ('P58154', '3wtj', '3u8j'), ('P58154', '3wtl', '3wtm'), ('P58154', '3wtm', '3u8l'), ('P58154', '3wtn', '1uw6'), ('P59071', '1fv0', '1sv3'), ('P59071', '1jq8', '1fv0'), ('P59071', '1kpm', '1jq8'), ('P59071', '1q7a', '2arm'), ('P59071', '1sv3', '1q7a'), ('P59071', '2arm', '1kpm'), ('P59071', '3h1x', '1kpm'), ('P60045', '1oxr', '1td7'), ('P60045', '1td7', '1oxr'), ('P61823', '1afk', '1o0h'), ('P61823', '1afl', '1afk'), ('P61823', '1jn4', '1jvu'), ('P61823', '1jvu', '1rnm'), ('P61823', '1o0f', '1o0h'), ('P61823', '1o0h', '1afk'), ('P61823', '1o0m', '1qhc'), ('P61823', '1o0n', '1afk'), ('P61823', '1qhc', '1o0h'), ('P61823', '1rnm', '1afl'), ('P61964', '4ql1', '6dar'), ('P61964', '5m23', '5sxm'), ('P61964', '5m25', '5sxm'), ('P61964', '5sxm', '6dar'), ('P61964', '6d9x', '5m25'), ('P61964', '6dai', '6dak'), ('P61964', '6dak', '6dai'), ('P61964', '6dar', '6dai'), ('P62508', '2p7a', '2p7g'), ('P62508', '2p7g', '2p7z'), ('P62617', '2amt', '3elc'), ('P62617', '2gzl', '3elc'), ('P62617', '3elc', '2amt'), ('P62937', '5t9u', '6gjj'), ('P62937', '5t9w', '5ta4'), ('P62937', '5t9z', '6gjm'), ('P62937', '6gji', '5t9w'), ('P62937', '6gjj', '5ta4'), ('P62937', '6gjl', '6gjm'), ('P62937', '6gjm', '5t9z'), ('P62937', '6gjn', '5t9z'), ('P62942', '1d7i', '1fkf'), ('P62942', '1d7j', '1fkb'), ('P62942', '1fkg', '1d7j'), ('P62942', '1fkh', '1fkf'), ('P62942', '1fki', '1fkf'), ('P62942', '1j4r', '1fkg'), ('P62993', '3ov1', '3s8n'), ('P62993', '3ove', '3ov1'), ('P62993', '3s8l', '3s8o'), ('P62993', '3s8n', '3ov1'), ('P62993', '3s8o', '3ove'), ('P63086', '4qyy', '6cpw'), ('P63086', '6cpw', '4qyy'), ('P64012', '3zhx', '3zi0'), ('P64012', '3zi0', '3zhx'), ('P66034', '2c92', '2c97'), ('P66034', '2c94', '2c97'), ('P66034', '2c97', '2c94'), ('P66992', '3qqs', '3r88'), ('P66992', '3r88', '4m0r'), ('P66992', '3twp', '3uu1'), ('P66992', '3uu1', '3twp'), ('P66992', '4gkm', '3qqs'), ('P66992', '4ij1', '4owv'), ('P66992', '4m0r', '4gkm'), ('P66992', '4n8q', '3twp'), ('P66992', '4owm', '4m0r'), ('P66992', '4owv', '3twp'), ('P68400', '2zjw', '5h8e'), ('P68400', '3bqc', '5cu4'), ('P68400', '3h30', '3pe2'), ('P68400', '3pe1', '5cu4'), ('P68400', '3pe2', '3pe1'), ('P68400', '5cqu', '5cu4'), ('P68400', '5csp', '3h30'), ('P68400', '5cu4', '3pe1'), ('P68400', '5h8e', '5csp'), ('P69834', '5mrm', '5mro'), ('P69834', '5mro', '5mrp'), ('P69834', '5mrp', '5mro'), ('P71094', '2jke', '2zq0'), ('P71094', '2jkp', '2zq0'), ('P71094', '2zq0', '2jkp'), ('P71447', '1z4o', '2wf5'), ('P71447', '2wf5', '1z4o'), ('P76141', '4l4z', '4l51'), ('P76141', '4l50', '4l51'), ('P76141', '4l51', '4l50'), ('P76637', '1ec9', '1ecq'), ('P76637', '1ecq', '1ec9'), ('P78536', '2oi0', '3l0v'), ('P78536', '3b92', '3le9'), ('P78536', '3ewj', '3le9'), ('P78536', '3kmc', '3le9'), ('P78536', '3l0v', '3lea'), ('P78536', '3le9', '3ewj'), ('P78536', '3lea', '3le9'), ('P80188', '3dsz', '3tf6'), ('P80188', '3tf6', '3dsz'), ('P84887', '2hjb', '2q7q'), ('P84887', '2q7q', '2hjb'), ('P95607', '3i4y', '3i51'), ('P95607', '3i51', '3i4y'), ('P96257', '6h1u', '6h2t'), ('P96257', '6h2t', '6h1u'), ('P98170', '2vsl', '4j46'), ('P98170', '3cm2', '3hl5'), ('P98170', '3hl5', '4j48'), ('P98170', '4j44', '4j45'), ('P98170', '4j45', '4j44'), ('P98170', '4j46', '3hl5'), ('P98170', '4j47', '4j44'), ('P98170', '4j48', '3cm2'), ('P9WMC0', '5f08', '5j3l'), ('P9WMC0', '5f0f', '5j3l'), ('P9WMC0', '5j3l', '5f08'), ('P9WPQ5', '6cvf', '6czc'), ('P9WPQ5', '6czb', '6cze'), ('P9WPQ5', '6czc', '6czb'), ('P9WPQ5', '6cze', '6czc'), ('Q00972', '3tz0', '4h7q'), ('Q00972', '4dzy', '4h85'), ('Q00972', '4h7q', '4h85'), ('Q00972', '4h81', '4h7q'), ('Q00972', '4h85', '4dzy'), ('Q00987', '4erf', '4wt2'), ('Q00987', '4hbm', '4mdn'), ('Q00987', '4mdn', '4wt2'), ('Q00987', '4wt2', '4mdn'), ('Q00987', '4zyf', '4hbm'), ('Q01693', '1ft7', '3b3w'), ('Q01693', '1igb', '1ft7'), ('Q01693', '1txr', '1igb'), ('Q01693', '3b3c', '3b7i'), ('Q01693', '3b3s', '3b3c'), ('Q01693', '3b3w', '1txr'), ('Q01693', '3b7i', '1igb'), ('Q01693', '3vh9', '3b3w'), ('Q03111', '6hpw', '6ht1'), ('Q03111', '6ht1', '6hpw'), ('Q04609', '2xef', '2xei'), ('Q04609', '2xei', '4ngp'), ('Q04609', '2xej', '3sjf'), ('Q04609', '3iww', '3rbu'), ('Q04609', '3rbu', '2xef'), ('Q04609', '3sjf', '4ngn'), ('Q04609', '4ngm', '4ngp'), ('Q04609', '4ngn', '2xeg'), ('Q04609', '4ngp', '2xeg'), ('Q04631', '1o1s', '1qbq'), ('Q04631', '1qbq', '1o1s'), ('Q05097', '2wyf', '4a6s'), ('Q05097', '3zyf', '4ljh'), ('Q05097', '4a6s', '4ljh'), ('Q05097', '4ljh', '3zyf'), ('Q05097', '4lk7', '2wyf'), ('Q05127', '4ibb', '4ibf'), ('Q05127', '4ibc', '4ibg'), ('Q05127', '4ibd', '4ibc'), ('Q05127', '4ibe', '4ibd'), ('Q05127', '4ibf', '4ibe'), ('Q05127', '4ibg', '4ibe'), ('Q05127', '4ibi', '4ibc'), ('Q05127', '4ibj', '4ibb'), ('Q05127', '4ibk', '4ibc'), ('Q05397', '4gu6', '4kao'), ('Q05397', '4gu9', '4kao'), ('Q05397', '4k9y', '4kao'), ('Q05397', '4kao', '4k9y'), ('Q06135', '5o9o', '5o9y'), ('Q06135', '5o9p', '5o9y'), ('Q06135', '5o9q', '5o9r'), ('Q06135', '5o9r', '5o9o'), ('Q06135', '5o9y', '5o9q'), ('Q06135', '5oa2', '5oa6'), ('Q06135', '5oa6', '5o9p'), ('Q06GJ0', '3nsn', '3s6t'), ('Q06GJ0', '3ozp', '3nsn'), ('Q06GJ0', '3s6t', '3ozp'), ('Q06GJ0', '3vtr', '3wmc'), ('Q06GJ0', '3wmc', '3vtr'), ('Q07075', '4kx8', '4kxb'), ('Q07075', '4kxb', '4kx8'), ('Q07343', '1ro6', '3o56'), ('Q07343', '3o56', '5laq'), ('Q07343', '5laq', '3o56'), ('Q07817', '2yxj', '3spf'), ('Q07817', '3qkd', '3zln'), ('Q07817', '3spf', '4c5d'), ('Q07817', '3zk6', '3zlr'), ('Q07817', '3zln', '3spf'), ('Q07817', '3zlr', '4c52'), ('Q07817', '4c52', '3zlr'), ('Q07817', '4c5d', '3spf'), ('Q07820', '4hw3', '6b4u'), ('Q07820', '4zbf', '6b4l'), ('Q07820', '4zbi', '6fs0'), ('Q07820', '5vkc', '6fs1'), ('Q07820', '6b4l', '4zbi'), ('Q07820', '6b4u', '6fs1'), ('Q07820', '6fs0', '6b4u'), ('Q07820', '6fs1', '6b4u'), ('Q08638', '1oif', '2j75'), ('Q08638', '1uz1', '2j75'), ('Q08638', '1w3j', '1uz1'), ('Q08638', '2cbu', '2j75'), ('Q08638', '2cbv', '2j78'), ('Q08638', '2ces', '2j78'), ('Q08638', '2cet', '1oif'), ('Q08638', '2j75', '2cbu'), ('Q08638', '2j77', '2cet'), ('Q08638', '2j78', '1oif'), ('Q08881', '3miy', '3qgw'), ('Q08881', '3qgw', '4qd6'), ('Q08881', '3qgy', '4qd6'), ('Q08881', '4m0y', '4m13'), ('Q08881', '4m12', '3qgw'), ('Q08881', '4m13', '4m14'), ('Q08881', '4m14', '4qd6'), ('Q08881', '4qd6', '3qgw'), ('Q08881', '4rfm', '4m0y'), ('Q0A480', '4ks1', '4ks4'), ('Q0A480', '4ks4', '4ks1'), ('Q0ED31', '4dkp', '4dko'), ('Q0ED31', '4dkq', '4dkp'), ('Q0ED31', '4dkr', '4dkq'), ('Q0ED31', '4i54', '4dkp'), ('Q0P8Q4', '5ad1', '5tcy'), ('Q0P8Q4', '5tcy', '5ad1'), ('Q0PBL7', '3fj7', '3fjg'), ('Q0PBL7', '3fjg', '3fj7'), ('Q0T8Y8', '4xoc', '4xoe'), ('Q0T8Y8', '4xoe', '4xoc'), ('Q0TR53', '2j62', '2x0y'), ('Q0TR53', '2wb5', '2xpk'), ('Q0TR53', '2x0y', '2wb5'), ('Q0TR53', '2xpk', '2j62'), ('Q10714', '1j36', '4ca7'), ('Q10714', '1j37', '2x95'), ('Q10714', '2x8z', '1j37'), ('Q10714', '2x91', '4ca8'), ('Q10714', '2x95', '2x97'), ('Q10714', '2x96', '2x91'), ('Q10714', '2x97', '1j37'), ('Q10714', '2xhm', '2x91'), ('Q10714', '4ca7', '2x95'), ('Q10714', '4ca8', '1j36'), ('Q12051', '2e91', '2e92'), ('Q12051', '2e92', '2e91'), ('Q12051', '2e94', '2e91'), ('Q12852', '5cep', '5vo1'), ('Q12852', '5ceq', '5vo1'), ('Q12852', '5vo1', '5ceq'), ('Q13093', '5lz4', '5lz5'), ('Q13093', '5lz5', '5lz4'), ('Q13093', '5lz7', '5lz4'), ('Q13133', '3ipu', '3ipq'), ('Q13153', '4zji', '5ime'), ('Q13153', '5dey', '4zji'), ('Q13153', '5dfp', '4zji'), ('Q13153', '5ime', '5dfp'), ('Q13451', '4jfk', '4w9o'), ('Q13451', '4jfm', '4w9p'), ('Q13451', '4w9o', '4jfk'), ('Q13451', '4w9p', '4jfk'), ('Q13451', '5dit', '4w9p'), ('Q13526', '2xp7', '3ikd'), ('Q13526', '3ikd', '3ikg'), ('Q13526', '3ikg', '2xp7'), ('Q13627', '6eif', '6eis'), ('Q13627', '6eij', '6eis'), ('Q13627', '6eiq', '6eir'), ('Q13627', '6eir', '6eij'), ('Q13627', '6eis', '6eiq'), ('Q14145', '4xmb', '5x54'), ('Q14145', '5x54', '4xmb'), ('Q14397', '4bb9', '4ly9'), ('Q14416', '4xaq', '4xas'), ('Q14416', '4xas', '4xaq'), ('Q15119', '4mpn', '5j6a'), ('Q15119', '5j6a', '4mpn'), ('Q15370', '4b9k', '4w9d'), ('Q15370', '4bks', '4b9k'), ('Q15370', '4bkt', '4w9h'), ('Q15370', '4w9c', '4w9h'), ('Q15370', '4w9d', '4w9k'), ('Q15370', '4w9f', '4b9k'), ('Q15370', '4w9h', '4w9j'), ('Q15370', '4w9i', '4w9k'), ('Q15370', '4w9j', '4w9i'), ('Q15370', '4w9k', '4w9f'), ('Q15562', '5dq8', '5dqe'), ('Q15562', '5dqe', '5dq8'), ('Q16539', '1kv1', '1yqj'), ('Q16539', '1yqj', '1kv1'), ('Q16539', '2baj', '3e92'), ('Q16539', '2bak', '3d7z'), ('Q16539', '2bal', '3d83'), ('Q16539', '2yix', '2zb1'), ('Q16539', '2zb1', '2yix'), ('Q16539', '3d7z', '2zb1'), ('Q16539', '3d83', '2bak'), ('Q16539', '3e92', '1kv1'), ('Q16769', '2afw', '2afx'), ('Q16769', '2afx', '2afw'), ('Q16769', '3pbb', '2afx'), ('Q18R04', '2h6b', '3e5u'), ('Q18R04', '3e5u', '2h6b'), ('Q1W640', '3s0b', '3s0d'), ('Q1W640', '3s0d', '3s0b'), ('Q1W640', '3s0e', '3s0d'), ('Q24451', '1ps3', '3d51'), ('Q24451', '2f7o', '1ps3'), ('Q24451', '2f7p', '3d51'), ('Q24451', '3d4y', '3d50'), ('Q24451', '3d4z', '3d52'), ('Q24451', '3d50', '3ddf'), ('Q24451', '3d51', '1ps3'), ('Q24451', '3d52', '3d50'), ('Q24451', '3ddf', '3ddg'), ('Q24451', '3ddg', '3ddf'), ('Q26998', '1jlr', '1upf'), ('Q26998', '1upf', '1jlr'), ('Q2A1P5', '4hy1', '4hym'), ('Q2A1P5', '4hym', '4hy1'), ('Q2PS28', '2pwd', '2pwg'), ('Q2PS28', '2pwg', '2pwd'), ('Q38BV6', '2ptz', '2pu1'), ('Q38BV6', '2pu1', '2ptz'), ('Q396C9', '3b4p', '3juo'), ('Q396C9', '3juo', '3jup'), ('Q396C9', '3jup', '3juo'), ('Q41931', '5gj9', '5gja'), ('Q41931', '5gja', '5gj9'), ('Q42975', '3f5j', '3f5k'), ('Q42975', '3f5k', '3f5l'), ('Q42975', '3f5l', '3f5k'), ('Q460N5', '5o2d', '3q71'), ('Q46822', '1q54', '2vnp'), ('Q46822', '2vnp', '1q54'), ('Q48255', '2xd9', '4b6s'), ('Q48255', '2xda', '4b6r'), ('Q48255', '4b6r', '2xd9'), ('Q48255', '4b6s', '2xd9'), ('Q4W803', '3ahn', '3aho'), ('Q4W803', '3aho', '3ahn'), ('Q52L64', '2r1y', '2r23'), ('Q52L64', '2r23', '2r1y'), ('Q54727', '2vw1', '2vw2'), ('Q54727', '2vw2', '2vw1'), ('Q57ZL6', '4i71', '4i72'), ('Q57ZL6', '4i72', '4i74'), ('Q57ZL6', '4i74', '4i72'), ('Q58597', '4rrf', '4rrg'), ('Q58597', '4rrg', '4rrf'), ('Q58EU8', '1yei', '2c1p'), ('Q58EU8', '2c1p', '1yei'), ('Q58F21', '4flp', '4kcx'), ('Q58F21', '4kcx', '4flp'), ('Q5A4W8', '5n17', '5n18'), ('Q5A4W8', '5n18', '5n17'), ('Q5G940', '2glp', '3b7j'), ('Q5G940', '3b7j', '3ed0'), ('Q5G940', '3cf8', '3b7j'), ('Q5G940', '3ed0', '3b7j'), ('Q5M4H8', '5oxk', '6ghj'), ('Q5M4H8', '6ghj', '5oxk'), ('Q5RZ08', '2r3w', '6cdl'), ('Q5RZ08', '2r43', '6dil'), ('Q5RZ08', '5wlo', '6cdj'), ('Q5RZ08', '6cdj', '2r3t'), ('Q5RZ08', '6cdl', '2r43'), ('Q5RZ08', '6dif', '5wlo'), ('Q5RZ08', '6dil', '6dj1'), ('Q5RZ08', '6dj1', '2r43'), ('Q5SH52', '1wuq', '1wur'), ('Q5SH52', '1wur', '1wuq'), ('Q5SID9', '1odi', '1odj'), ('Q5SID9', '1odj', '1odi'), ('Q63226', '2v3u', '5cc2'), ('Q63226', '5cc2', '2v3u'), ('Q6DPL2', '3ckz', '3cl0'), ('Q6DPL2', '3cl0', '3ckz'), ('Q6G8R1', '4ai5', '4aia'), ('Q6G8R1', '4aia', '4ai5'), ('Q6N193', '5oei', '5oku'), ('Q6N193', '5oku', '5oei'), ('Q6P6C2', '4o61', '4oct'), ('Q6P6C2', '4oct', '4o61'), ('Q6PL18', '4qsu', '4tu4'), ('Q6PL18', '4qsv', '4tte'), ('Q6PL18', '4tt2', '4qsv'), ('Q6PL18', '4tte', '4tz2'), ('Q6PL18', '4tu4', '4qsu'), ('Q6PL18', '4tz2', '4qsu'), ('Q6PL18', '5a5q', '5a81'), ('Q6PL18', '5a81', '4tt2'), ('Q6R308', '3f7h', '3f7i'), ('Q6R308', '3f7i', '3gta'), ('Q6R308', '3gt9', '3f7i'), ('Q6R308', '3gta', '3f7h'), ('Q6T755', '3uj9', '3ujc'), ('Q6T755', '3ujc', '3uj9'), ('Q6T755', '3ujd', '3ujc'), ('Q6WVP6', '4q3t', '4q3u'), ('Q6WVP6', '4q3u', '4q3t'), ('Q6XEC0', '5qa8', '5qal'), ('Q6XEC0', '5qal', '5qay'), ('Q6XEC0', '5qay', '5qal'), ('Q70I53', '5g17', '5g1a'), ('Q70I53', '5g1a', '5g17'), ('Q72498', '3ao2', '3ao5'), ('Q72498', '3ao4', '3ovn'), ('Q72498', '3ao5', '3ao2'), ('Q72498', '3ovn', '3ao5'), ('Q75I93', '4qlk', '4qll'), ('Q75I93', '4qll', '4qlk'), ('Q76353', '3zso', '4ceb'), ('Q76353', '3zsq', '3zso'), ('Q76353', '3zsx', '3zt2'), ('Q76353', '3zsy', '4ceb'), ('Q76353', '3zt2', '4cgi'), ('Q76353', '3zt3', '4ceb'), ('Q76353', '4ceb', '4cig'), ('Q76353', '4cgi', '4cj4'), ('Q76353', '4cig', '3zsx'), ('Q76353', '4cj4', '3zsq'), ('Q7B8P6', '3qps', '3qqa'), ('Q7B8P6', '3qqa', '3qps'), ('Q7CX36', '3ip5', '3ip9'), ('Q7CX36', '3ip6', '3ip9'), ('Q7CX36', '3ip9', '3ip6'), ('Q7D2F4', '4ra1', '4zeb'), ('Q7D2F4', '4zeb', '4zei'), ('Q7D2F4', '4zec', '4zei'), ('Q7D2F4', '4zei', '4zec'), ('Q7D447', '5l9o', '5lom'), ('Q7D447', '5lom', '5l9o'), ('Q7D737', '2bes', '2bet'), ('Q7D737', '2bet', '2bes'), ('Q7D785', '3rv8', '3veh'), ('Q7D785', '3veh', '3rv8'), ('Q7DDU0', '4ipi', '4ipj'), ('Q7DDU0', '4ipj', '4ipi'), ('Q7SSI0', '3s54', '4kb9'), ('Q7SSI0', '4kb9', '3s54'), ('Q7ZCI0', '6dh1', '6dh2'), ('Q7ZCI0', '6dh2', '6dh1'), ('Q81R22', '4elf', '4elg'), ('Q81R22', '4elg', '4elf'), ('Q81R22', '4elh', '4elf'), ('Q86C09', '2i19', '3egt'), ('Q86C09', '3egt', '2i19'), ('Q86U86', '5fh7', '5ii2'), ('Q86U86', '5fh8', '5hrv'), ('Q86U86', '5hrv', '5hrw'), ('Q86U86', '5hrw', '5hrv'), ('Q86U86', '5hrx', '5fh7'), ('Q86U86', '5ii2', '5fh7'), ('Q86WV6', '4loi', '4qxo'), ('Q86WV6', '4qxo', '4ksy'), ('Q873X9', '1w9u', '1w9v'), ('Q873X9', '1w9v', '2iuz'), ('Q873X9', '2iuz', '1w9v'), ('Q89ZI2', '2j47', '2w4x'), ('Q89ZI2', '2j4g', '2w67'), ('Q89ZI2', '2jiw', '2w66'), ('Q89ZI2', '2vvn', '2xj7'), ('Q89ZI2', '2vvs', '2wca'), ('Q89ZI2', '2w4x', '2w67'), ('Q89ZI2', '2w66', '2j4g'), ('Q89ZI2', '2w67', '2jiw'), ('Q89ZI2', '2wca', '2jiw'), ('Q89ZI2', '2xj7', '2j47'), ('Q8A0N1', '2wvz', '2wzs'), ('Q8A0N1', '2wzs', '2wvz'), ('Q8A3I4', '2xib', '4j28'), ('Q8A3I4', '2xii', '4pcs'), ('Q8A3I4', '4j28', '4pee'), ('Q8A3I4', '4jfs', '2wvt'), ('Q8A3I4', '4pcs', '4jfs'), ('Q8A3I4', '4pee', '2wvt'), ('Q8AAK6', '2vjx', '2vot'), ('Q8AAK6', '2vl4', '2vjx'), ('Q8AAK6', '2vmf', '2vqt'), ('Q8AAK6', '2vo5', '2vl4'), ('Q8AAK6', '2vot', '2vjx'), ('Q8AAK6', '2vqt', '2vot'), ('Q8I3X4', '1nw4', '1q1g'), ('Q8I3X4', '1q1g', '1nw4'), ('Q8II92', '2y8c', '3t64'), ('Q8II92', '3t60', '3t64'), ('Q8II92', '3t64', '3t70'), ('Q8II92', '3t70', '2y8c'), ('Q8N1Q1', '3czv', '4hu1'), ('Q8N1Q1', '4hu1', '4knm'), ('Q8N1Q1', '4knm', '4qjx'), ('Q8N1Q1', '4knn', '4qjx'), ('Q8N1Q1', '4qjx', '4knm'), ('Q8Q3H0', '2cej', '4a6b'), ('Q8Q3H0', '2cen', '4a4q'), ('Q8Q3H0', '4a4q', '2cej'), ('Q8Q3H0', '4a6b', '4a6c'), ('Q8Q3H0', '4a6c', '5dgu'), ('Q8Q3H0', '5dgu', '4a4q'), ('Q8Q3H0', '5dgw', '5dgu'), ('Q8TEK3', '3qox', '3sr4'), ('Q8TEK3', '3sr4', '3qox'), ('Q8TEK3', '4ek9', '3qox'), ('Q8TF76', '6g34', '6g3a'), ('Q8TF76', '6g35', '6g3a'), ('Q8TF76', '6g36', '6g38'), ('Q8TF76', '6g37', '6g35'), ('Q8TF76', '6g38', '6g34'), ('Q8TF76', '6g39', '6g3a'), ('Q8TF76', '6g3a', '6g35'), ('Q8ULI3', '6dh6', '6dh8'), ('Q8ULI3', '6dh7', '6dh6'), ('Q8ULI3', '6dh8', '6dh7'), ('Q8VPB3', '2vpn', '2vpo'), ('Q8VPB3', '2vpo', '2vpn'), ('Q8WQX9', '5g2b', '5l8y'), ('Q8WQX9', '5g57', '5l8y'), ('Q8WQX9', '5g5v', '5l8c'), ('Q8WQX9', '5l8c', '5g57'), ('Q8WQX9', '5l8y', '5l8c'), ('Q8WSF8', '2byr', '2xys'), ('Q8WSF8', '2bys', '2wnj'), ('Q8WSF8', '2pgz', '2ymd'), ('Q8WSF8', '2wn9', '2wnj'), ('Q8WSF8', '2wnc', '2wn9'), ('Q8WSF8', '2wnj', '2wn9'), ('Q8WSF8', '2x00', '2wn9'), ('Q8WSF8', '2xys', '2x00'), ('Q8WSF8', '2xyt', '2wn9'), ('Q8WSF8', '2ymd', '2x00'), ('Q8WUI4', '3znr', '3zns'), ('Q8WUI4', '3zns', '3znr'), ('Q8WXF7', '4idn', '4ido'), ('Q8WXF7', '4ido', '4idn'), ('Q8XXK6', '2bt9', '4csd'), ('Q8XXK6', '4csd', '2bt9'), ('Q8Y8D7', '2i2c', '4dy6'), ('Q8Y8D7', '4dy6', '5dhu'), ('Q8Y8D7', '5dhu', '4dy6'), ('Q90EB9', '1izh', '1izi'), ('Q90EB9', '1izi', '1izh'), ('Q90JJ9', '4m8x', '4m8y'), ('Q90JJ9', '4m8y', '4m8x'), ('Q90K99', '3o99', '4djr'), ('Q90K99', '3o9a', '3o99'), ('Q90K99', '3o9d', '3o99'), ('Q90K99', '3o9e', '4djp'), ('Q90K99', '3o9i', '3o99'), ('Q90K99', '4djo', '3o9i'), ('Q90K99', '4djp', '3o9i'), ('Q90K99', '4djq', '3o9a'), ('Q90K99', '4djr', '3o9e'), ('Q92769', '4lxz', '5iwg'), ('Q92769', '4ly1', '5ix0'), ('Q92769', '5iwg', '4ly1'), ('Q92769', '5ix0', '4ly1'), ('Q92793', '4tqn', '5h85'), ('Q92793', '4yk0', '5mme'), ('Q92793', '5dbm', '5eng'), ('Q92793', '5eng', '5h85'), ('Q92793', '5ep7', '5i8g'), ('Q92793', '5h85', '5ep7'), ('Q92793', '5i8g', '5j0d'), ('Q92793', '5j0d', '5dbm'), ('Q92793', '5mme', '5dbm'), ('Q92793', '5mmg', '5h85'), ('Q92831', '5fe6', '5lvq'), ('Q92831', '5fe7', '5lvr'), ('Q92831', '5fe9', '5fe7'), ('Q92831', '5lvq', '5fe7'), ('Q92831', '5lvr', '5lvq'), ('Q92N37', '2reg', '2rin'), ('Q92N37', '2rin', '2reg'), ('Q92WC8', '2q88', '2q89'), ('Q92WC8', '2q89', '2q88'), ('Q939R9', '3str', '3sw8'), ('Q939R9', '3sw8', '3str'), ('Q93UV0', '2w8j', '2w8w'), ('Q93UV0', '2w8w', '2w8j'), ('Q95NY5', '3nhi', '3nht'), ('Q95NY5', '3nht', '3nhi'), ('Q96CA5', '2i3h', '2i3i'), ('Q96CA5', '2i3i', '3f7g'), ('Q96CA5', '3f7g', '3uw5'), ('Q96CA5', '3uw5', '2i3h'), ('Q980A5', '4rd0', '4rd3'), ('Q980A5', '4rd3', '4rd6'), ('Q980A5', '4rd6', '4rd3'), ('Q99640', '5vcv', '5vcy'), ('Q99640', '5vcw', '5vd1'), ('Q99640', '5vcy', '5vd0'), ('Q99640', '5vcz', '5vd3'), ('Q99640', '5vd0', '5vd1'), ('Q99640', '5vd1', '5vcy'), ('Q99640', '5vd3', '5vd0'), ('Q99814', '4ghi', '5tbm'), ('Q99814', '5tbm', '5ufp'), ('Q99814', '5ufp', '5tbm'), ('Q99AU2', '2d3u', '2d3z'), ('Q99AU2', '2d3z', '2d3u'), ('Q99QC1', '5k1d', '5k1f'), ('Q99QC1', '5k1f', '5k1d'), ('Q9AMP1', '3arp', '3arx'), ('Q9AMP1', '3arq', '3arw'), ('Q9AMP1', '3arw', '3arp'), ('Q9AMP1', '3arx', '3arp'), ('Q9BJF5', '3sxf', '3v5p'), ('Q9BJF5', '3t3u', '3v5t'), ('Q9BJF5', '3v51', '3v5t'), ('Q9BJF5', '3v5p', '3v51'), ('Q9BJF5', '3v5t', '3v5p'), ('Q9BY41', '3mz6', '5vi6'), ('Q9BY41', '5vi6', '3mz6'), ('Q9BZP6', '3rm4', '3rm9'), ('Q9BZP6', '3rm9', '3rm4'), ('Q9CPU0', '4kyh', '4kyk'), ('Q9CPU0', '4kyk', '4pv5'), ('Q9CPU0', '4pv5', '4kyk'), ('Q9E7M1', '3sm2', '3slz'), ('Q9F4L3', '3d7k', '3iae'), ('Q9F4L3', '3iae', '3d7k'), ('Q9FUZ2', '3m6r', '3pn4'), ('Q9FUZ2', '3pn4', '3m6r'), ('Q9FV53', '4je7', '4je8'), ('Q9FV53', '4je8', '4je7'), ('Q9GK12', '3ng4', '3nw3'), ('Q9GK12', '3nw3', '3ng4'), ('Q9GK12', '3o4k', '3ng4'), ('Q9GK12', '3usx', '3ng4'), ('Q9GK12', '4fnn', '3o4k'), ('Q9H2K2', '3kr8', '4j21'), ('Q9H2K2', '4iue', '4j22'), ('Q9H2K2', '4j21', '4kzq'), ('Q9H2K2', '4j22', '4j3l'), ('Q9H2K2', '4j3l', '3kr8'), ('Q9H2K2', '4kzq', '4j21'), ('Q9H2K2', '4kzu', '4j22'), ('Q9H8M2', '4xy8', '5ji8'), ('Q9H8M2', '5eu1', '5i7y'), ('Q9H8M2', '5f1h', '5ji8'), ('Q9H8M2', '5f25', '5igm'), ('Q9H8M2', '5f2p', '5f25'), ('Q9H8M2', '5i7x', '5f25'), ('Q9H8M2', '5i7y', '5f25'), ('Q9H8M2', '5igm', '5i7y'), ('Q9H8M2', '5ji8', '5f25'), ('Q9H9B1', '5tuz', '5vsf'), ('Q9H9B1', '5vsf', '5tuz'), ('Q9HGR1', '4ymg', '4ymh'), ('Q9HGR1', '4ymh', '4ymg'), ('Q9HPW4', '2cc7', '2ccc'), ('Q9HPW4', '2ccb', '2cc7'), ('Q9HPW4', '2ccc', '2ccb'), ('Q9HYN5', '2boj', '2jdm'), ('Q9HYN5', '2jdm', '2boj'), ('Q9HYN5', '2jdp', '2boj'), ('Q9HYN5', '2jdu', '2jdm'), ('Q9HYN5', '3zdv', '2jdm'), ('Q9JLU4', '5ovc', '5ovp'), ('Q9JLU4', '5ovp', '6exj'), ('Q9JLU4', '6exj', '5ovp'), ('Q9K169', '4uc5', '4uma'), ('Q9K169', '4uma', '4umb'), ('Q9K169', '4umb', '4uma'), ('Q9K169', '4umc', '4uc5'), ('Q9KS12', '3eeb', '3fzy'), ('Q9KS12', '3fzy', '3eeb'), ('Q9KU37', '2oxn', '3gs6'), ('Q9KU37', '3gs6', '2oxn'), ('Q9KU37', '3gsm', '2oxn'), ('Q9KWT6', '1y3n', '1y3p'), ('Q9KWT6', '1y3p', '1y3n'), ('Q9L5C8', '3g2y', '3g31'), ('Q9L5C8', '3g2z', '3g30'), ('Q9L5C8', '3g30', '3g31'), ('Q9L5C8', '3g31', '3g34'), ('Q9L5C8', '3g32', '3g35'), ('Q9L5C8', '3g34', '4de0'), ('Q9L5C8', '3g35', '3g32'), ('Q9L5C8', '4de0', '3g34'), ('Q9L5C8', '4de1', '3g30'), ('Q9L5C8', '4de2', '3g34'), ('Q9N1E2', '1g98', '1koj'), ('Q9N1E2', '1koj', '1g98'), ('Q9NPB1', '1q91', '6g2l'), ('Q9NPB1', '6g2l', '6g2m'), ('Q9NPB1', '6g2m', '1q91'), ('Q9NQG6', '4nxu', '4nxv'), ('Q9NQG6', '4nxv', '4nxu'), ('Q9NR97', '4r0a', '5wyz'), ('Q9NR97', '5wyx', '4r0a'), ('Q9NR97', '5wyz', '5wyx'), ('Q9NXS2', '3pb7', '3pb9'), ('Q9NXS2', '3pb8', '3pb9'), ('Q9NXS2', '3pb9', '3pb8'), ('Q9NY33', '3t6b', '5e3a'), ('Q9NY33', '5e3a', '3t6b'), ('Q9NZD2', '2euk', '2evl'), ('Q9NZD2', '2evl', '2euk'), ('Q9PTT3', '6gnm', '6gnp'), ('Q9PTT3', '6gnp', '6gnw'), ('Q9PTT3', '6gnr', '6gnw'), ('Q9PTT3', '6gnw', '6gon'), ('Q9PTT3', '6gon', '6gnm'), ('Q9QB59', '3lzs', '3lzu'), ('Q9QB59', '3lzu', '3lzs'), ('Q9QLL6', '5efa', '5efc'), ('Q9QLL6', '5efc', '5efa'), ('Q9QYJ6', '2o8h', '2ovv'), ('Q9QYJ6', '2ovv', '2ovy'), ('Q9QYJ6', '2ovy', '2o8h'), ('Q9R0G6', '3v2n', '3v2q'), ('Q9R0G6', '3v2p', '3v2n'), ('Q9R0G6', '3v2q', '3v2p'), ('Q9R4E4', '2pqb', '2pqc'), ('Q9R4E4', '2pqc', '2pqb'), ('Q9RA63', '4lj5', '4lj8'), ('Q9RA63', '4lj8', '4lj5'), ('Q9RS96', '5hva', '5hwu'), ('Q9RS96', '5hwu', '5hva'), ('Q9T0I8', '2qtg', '3lgs'), ('Q9T0I8', '2qtt', '3lgs'), ('Q9T0I8', '3lgs', '2qtt'), ('Q9U9J6', '3cyz', '3d78'), ('Q9U9J6', '3cz1', '3cyz'), ('Q9U9J6', '3d78', '3cz1'), ('Q9UBN7', '5kh3', '6ce6'), ('Q9UBN7', '6ce6', '5kh3'), ('Q9UBN7', '6ced', '5kh3'), ('Q9UGN5', '3kjd', '4zzx'), ('Q9UGN5', '4zzx', '4zzy'), ('Q9UGN5', '4zzy', '4zzx'), ('Q9UIF8', '4nra', '5e73'), ('Q9UIF8', '4rvr', '5mge'), ('Q9UIF8', '5e73', '5e74'), ('Q9UIF8', '5e74', '5mgf'), ('Q9UIF8', '5mge', '5e73'), ('Q9UIF9', '5mgj', '6fgg'), ('Q9UIF9', '5mgk', '5mgj'), ('Q9UM73', '2xb7', '5fto'), ('Q9UM73', '4cd0', '5aa9'), ('Q9UM73', '4clj', '4cmo'), ('Q9UM73', '4cmo', '2xb7'), ('Q9UM73', '5aa9', '5kz0'), ('Q9UM73', '5fto', '2xb7'), ('Q9UM73', '5kz0', '5aa9'), ('Q9VHA0', '2r58', '2r5a'), ('Q9VHA0', '2r5a', '2r58'), ('Q9VWX8', '5aan', '5fyx'), ('Q9VWX8', '5fyx', '6epa'), ('Q9VWX8', '6epa', '5aan'), ('Q9WUL6', '5t8o', '5t8p'), ('Q9WUL6', '5t8p', '5t8o'), ('Q9WYE2', '2zwz', '2zx8'), ('Q9WYE2', '2zx6', '2zwz'), ('Q9WYE2', '2zx7', '2zwz'), ('Q9WYE2', '2zx8', '2zx6'), ('Q9XEI3', '1x38', '1x39'), ('Q9XEI3', '1x39', '1x38'), ('Q9Y233', '3ui7', '4lm0'), ('Q9Y233', '3uuo', '4lkq'), ('Q9Y233', '4dff', '3ui7'), ('Q9Y233', '4hf4', '3uuo'), ('Q9Y233', '4lkq', '4lm0'), ('Q9Y233', '4llj', '4dff'), ('Q9Y233', '4llk', '3ui7'), ('Q9Y233', '4llp', '4llj'), ('Q9Y233', '4llx', '4llk'), ('Q9Y233', '4lm0', '4lkq'), ('Q9Y3Q0', '3fed', '3fee'), ('Q9Y3Q0', '3fee', '3ff3'), ('Q9Y3Q0', '3ff3', '3fee'), ('Q9Y5Y6', '2gv6', '2gv7'), ('Q9Y5Y6', '2gv7', '4jyt'), ('Q9Y5Y6', '4jyt', '4o9v'), ('Q9Y5Y6', '4jzi', '2gv7'), ('Q9Y5Y6', '4o97', '4jz1'), ('Q9Y657', '4h75', '5jsj'), ('Q9Y657', '5jsg', '5jsj'), ('Q9Y657', '5jsj', '5jsg'), ('Q9Y6F1', '3c4h', '3fhb'), ('Q9Y6F1', '3fhb', '3c4h'), ('Q9YFY3', '4rr6', '4rra'), ('Q9YFY3', '4rra', '4rr6'), ('Q9Z2X8', '5fnr', '5fnt'), ('Q9Z2X8', '5fns', '5fnt'), ('Q9Z2X8', '5fnt', '5fnu'), ('Q9Z2X8', '5fnu', '5fnr'), ('Q9Z4P9', '4d4d', '5n0f'), ('Q9Z4P9', '5m77', '4d4d'), ('Q9Z4P9', '5n0f', '4d4d'), ('Q9ZMY2', '4ffs', '4wkn'), ('Q9ZMY2', '4wkn', '4ynb'), ('Q9ZMY2', '4wko', '4ynb'), ('Q9ZMY2', '4wkp', '4ynb'), ('Q9ZMY2', '4ynb', '4wkn'), ('Q9ZMY2', '4yo8', '4wkn'), ('U5XBU0', '4cpy', '4cpz'), ('U5XBU0', '4cpz', '4cpy'), ('U6NCW5', '4ovf', '4ovh'), ('U6NCW5', '4ovg', '4ovh'), ('U6NCW5', '4ovh', '4ovg'), ('U6NCW5', '4pnu', '4ovf'), ('V5Y949', '4q1w', '4q1y'), ('V5Y949', '4q1x', '4q1y'), ('V5Y949', '4q1y', '4q1w'), ('V5YAB1', '5kr1', '5kr2'), ('V5YAB1', '5kr2', '5kr1'), ('W5R8B8', '5nze', '5nzn'), ('W5R8B8', '5nzf', '5nzn'), ('W5R8B8', '5nzn', '5nze')]
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