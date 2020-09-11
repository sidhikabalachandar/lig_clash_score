"""
The purpose of this code is to calculate the mcss of every target/starting ligand pair

It can be run on sherlock using
$ ml load chemistry
$ ml load schrodinger
$ $SCHRODINGER/run python3 mcss_and_combine.py all /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
$ $SCHRODINGER/run python3 mcss_and_combine.py group /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw --index 0
$ $SCHRODINGER/run python3 mcss_and_combine.py check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
$ $SCHRODINGER/run python3 mcss_and_combine.py group_name_check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt /home/users/sidhikab/lig_clash_score/src/data/run /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw --index 0
$ $SCHRODINGER/run python3 mcss_similarity.py MAPK14
"""

import os
from schrodinger.structure import StructureReader, StructureWriter
import argparse
from tqdm import tqdm
import pickle

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
            with StructureReader(new_l1_path) as ligand1, StructureReader(new_l2_path) as ligand2:
                ligands = {l1: next(ligand1), l2: next(ligand2)}
                mcss.compute_mcss(ligands, init_file, mcss_types_file)
            os.system('rm {} {}'.format(new_l1_path, new_l2_path))

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

def run_all(grouped_files, docked_prot_file, run_path, raw_root):
    for i, group in enumerate(grouped_files):
        cmd = 'sbatch -p owners -t 1:00:00 -o {} --wrap="$SCHRODINGER/run python3 mcss_and_combine.py group {} {} {} ' \
              '--index {}"'
        os.system(cmd.format(os.path.join(run_path, 'mcss{}.out'.format(i)), docked_prot_file, run_path, raw_root, i))

def run_group(grouped_files, raw_root, index):
    for protein, target, start in grouped_files[index]:
        pair = '{}-to-{}'.format(target, start)
        protein_path = os.path.join(raw_root, protein)
        pair_path = os.path.join(protein_path, pair)
        pose_path = os.path.join(pair_path, 'ligand_poses')

        # comput mcss
        compute_protein_mcss([target, start], pair_path)

        # combine ligands
        with StructureWriter('{}/{}_merge_pv.mae'.format(pair_path, pair)) as all_file:
            for file in os.listdir(pose_path):
                if file[-3:] == 'mae':
                    pv = list(StructureReader(os.path.join(pose_path, file)))
                    all_file.append(pv[0])

        # zip file
        os.system('gzip -f {}/{}_merge_pv.mae'.format(pair_path, pair, pair_path, pair))

def run_check(docked_prot_file, raw_root):
    process = []
    num_pairs = 0
    with open(docked_prot_file) as fp:
        for line in tqdm(fp, desc='going through protein, target, start groups'):
            if line[0] == '#': continue
            protein, target, start = line.strip().split()
            pair = '{}-to-{}'.format(target, start)
            num_pairs += 1
            protein_path = os.path.join(raw_root, protein)
            pair_path = os.path.join(protein_path, pair)
            if not os.path.exists(os.path.join(pair_path, '{}_mcss.csv'.format(pair))):
                process.append((protein, target, start))
                continue
            if not os.path.exists(os.path.join(pair_path, '{}/{}_merge_pv.mae.gz'.format(pair_path, pair))):
                process.append((protein, target, start))
                continue

    print('Missing', len(process), '/', num_pairs)
    print(process)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either all, group, check, update, or MAPK14')
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('run_path', type=str, help='directory where script and output files will be written')
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('--index', type=int, default=-1, help='for group task, group number')
    parser.add_argument('--new_prot_file', type=str, default=os.path.join(os.getcwd(), 'index.txt'),
                        help='for update task, name of new prot file')
    parser.add_argument('--n', type=int, default=3, help='number of protein, target, start groups processed in '
                                                         'group task')
    parser.add_argument('--name_dir', type=str, default=os.path.join(os.getcwd(), 'names'),
                        help='for all_name_check and group_name_check task, directiory to place unfinished protein, '
                             'target, start groups')
    args = parser.parse_args()

    if not os.path.exists(args.run_path):
        os.mkdir(args.run_path)

    if args.task == 'all':
        process = get_prots(args.docked_prot_file)
        grouped_files = group_files(args.n, process)
        run_all(grouped_files, args.docked_prot_file, args.run_path, args.raw_root)

    if args.task == 'group':
        process = get_prots(args.docked_prot_file)
        grouped_files = group_files(args.n, process)
        run_group(grouped_files, args.raw_root, args.index)

    if args.task == 'check':
        run_check(args.docked_prot_file, args.raw_root)

    if args.task == 'all_name_check':
        if not os.path.exists(args.name_dir):
            os.mkdir(args.name_dir)

        process = get_prots(args.docked_prot_file)
        grouped_files = group_files(args.n, process)

        if not os.path.exists(args.run_path):
            os.mkdir(args.run_path)

        for i, group in enumerate(grouped_files):
            cmd = 'sbatch -p owners -t 0:20:00 -o {} --wrap="$SCHRODINGER/run python3 score_and_rmsd.py group_name_check ' \
                  '{} {} {} --index {}"'
            os.system(cmd.format(os.path.join(args.run_path, 'name{}.out'.format(i)), args.docked_prot_file,
                                 args.run_path, args.raw_root, i))

    if args.task == 'group_name_check':
        if not os.path.exists(args.name_dir):
            os.mkdir(args.name_dir)

        process = get_prots(args.docked_prot_file)
        grouped_files = group_files(args.n, process)
        unfinished = []

        for protein, target, start in grouped_files[args.index]:
            protein_path = os.path.join(args.raw_root, protein)
            pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
            pose_path = os.path.join(pair_path, 'ligand_poses')
            pv_file = os.path.join(pair_path, '{}-to-{}_merge_pv.mae'.format(target, start))
            files = []
            with open(pv_file, 'r') as f:
                for line in f:
                    if '{}_lig'.format(target) in line.strip() and '.mae' in line.strip():
                        files.append(line.strip())

            if len(files) != len(os.listdir(pose_path)):
                print(len(files), len(os.listdir(pose_path)))
                unfinished.append((protein, target, start))
                for i in range(100):
                    if '{}_lig{}.mae'.format(target, i) not in files:
                        print('{}_lig{}.mae'.format(target, i))
                    for j in range(10):
                        if '{}_lig{}.mae'.format(target, str(i) + chr(ord('a')+j)) not in files:
                            print('{}_lig{}.mae'.format(target, str(i) + chr(ord('a')+j)))

        # outfile = open(os.path.join(args.name_dir, '{}.pkl'.format(args.index)), 'wb')
        # pickle.dump(unfinished, outfile)
        print(unfinished)

    if args.task == 'check_name_check':
        process = get_prots(args.docked_prot_file)
        grouped_files = group_files(args.n, process)

        if len(os.listdir(args.name_dir)) != len(grouped_files):
            print('Not all files created')
        else:
            print('All files created')

        errors = []
        for i in range(len(grouped_files)):
            infile = open(os.path.join(args.name_dir, '{}.pkl'.format(i)), 'rb')
            unfinished = pickle.load(infile)
            infile.close()
            errors.extend(unfinished)

        print('Errors', len(errors), '/', len(process))
        print(errors)

    if args.task == 'update':
        text = []
        with open(args.docked_prot_file) as fp:
            for line in tqdm(fp, desc='files'):
                if line[0] == '#': continue
                protein, target, start = line.strip().split()
                protein_path = os.path.join(args.raw_root, protein)
                pair_path = os.path.join(protein_path, '{}-to-{}'.format(target, start))
                if os.path.exists(os.path.join(pair_path, '{}-to-{}_mcss.csv'.format(target, start))):
                    text.append(line)

        file = open(args.new_prot_file, "w")
        file.writelines(text)
        file.close()

if __name__ == '__main__':
    main()