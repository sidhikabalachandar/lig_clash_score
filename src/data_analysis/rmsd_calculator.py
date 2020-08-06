'''
This protocol can be used to find the rmsd between the residues in the binding pocket of every pair of structures of a protein
Only the residues within 4 angstroms of either structures' ligands are considered

# how to run this file:
# ml load chemistry
# ml load schrodinger
# $SCHRODINGER/run python3 rmsd_calculator.py all
# $SCHRODINGER/run python3 rmsd_calculator.py group alignment<index>.pkl
'''

import schrodinger.structure as structure
import schrodinger.structutils.measure as measure
import schrodinger.structutils.rmsd as rmsd
import os
import pickle
import sys

align_path = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/alignments'
chain_path = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/chains.pkl'
data_root = '/oak/stanford/groups/rondror/projects/ligand-docking/pdbbind_2019/data'
run_path = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/rmsd/run'
save_path = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/rmsd/rmsd{}.pkl'

'''
This function gets the pdbcode, chain, resnum, and inscode of every residue in the protein structure
It ignores any residues associated with the ligand
:param s: the protein structure 
:return: the list of every residue's pdbcode, chain, resnum, and inscode
'''


def get_all_res(s, chains, protein):
	r_list = []
	for m in list(s.molecule):
		for r in list(m.residue):
			if list(r.atom)[0].chain == chains[protein]:
				r_list.append((list(r.atom)[0].pdbcode, list(r.atom)[0].chain, list(r.atom)[0].resnum,
							   list(r.atom)[0].inscode))
	return r_list


'''
Maps unique residue identifiers to list index in alignment string

:param alignment_string: (string) output from alignment program, contains one letter codes and dashes
	example: 'TE--S--T-'
:param r_list: list of unique identifiers of each residue in order of sequence
	number of residues in r_list must be equal to number of residues in alignment_string
:return: the map of residues to alignment_string index
'''


def map_residues_to_align_index(alignment_string, r_list):
	r_to_i_map = {}
	counter = 0
	for i in range(len(alignment_string)):
		if counter >= len(r_list):
			break
		if alignment_string[i] == r_list[counter][0]:
			r_to_i_map[r_list[counter]] = i
			counter += 1
	return r_to_i_map


'''
This function inverses an input map
The keys become values and vice versa
:param m: the map
:return: the inversed map
'''


def inv_map(m):
	return {v: k for k, v in m.items()}


'''
This function gets the unique identifier for all residues within 4 angstroms of the ligand
:param s: the protein structure
:param r_to_i_map: the map of residues to alignment_string index
:return: a list of information for all residues within 4 angstroms of the ligand
'''


def get_res_near_ligand(r_to_i_map, pocket_s):
	close_r_set = set({})
	for m in list(pocket_s.molecule):
		for r in list(m.residue):
			if (list(r.atom)[0].pdbcode, list(r.atom)[0].chain, list(r.atom)[0].resnum, list(r.atom)[0].inscode) in r_to_i_map:
				close_r_set.add((list(r.atom)[0].pdbcode, list(r.atom)[0].chain, list(r.atom)[0].resnum, list(r.atom)[0].inscode))
	return close_r_set


'''
This function gets the atom list corresponding to a given list of unique residue identifiers from a given protein structure
:param s: the protein structure
:param final_r_list: the list of residues being compared between the two protein structures
:return: a list of ASL values for each residue,
		 a list of atoms for each residue,
		 a list of the backbone atom for each residue
		 a list of sidechain atoms for each residue
'''


def get_atoms(s, final_r_list):
	asl_list = []
	a_list = []

	for m in list(s.molecule):
		for r in list(m.residue):
			if (list(r.atom)[0].pdbcode, list(r.atom)[0].chain, list(r.atom)[0].resnum,
				list(r.atom)[0].inscode) in final_r_list:
				asl_list.append(r.getAsl())
				a_list.append(r.getAtomList())
	return (asl_list, a_list)


'''
find the rmsd between the residues in the binding pocket of every pair of structures of a protein
:param protein: name of the protein
:param rmsd_file: path to save location of rmsd_file
:param combind_root: path to the combind root folder
:return: 
'''


def compute_protein_rmsds(paired_strs, protein, start, target, s1, s2, chains, protein_folder):
	(paired_str_s1, paired_str_s2) = paired_strs[protein][start][target]

	r_list_s1 = get_all_res(s1, chains, protein)
	r_list_s2 = get_all_res(s2, chains, protein)

	r_to_i_map_s1 = map_residues_to_align_index(paired_str_s1, r_list_s1)
	r_to_i_map_s2 = map_residues_to_align_index(paired_str_s2, r_list_s2)
	i_to_r_map_s1 = inv_map(r_to_i_map_s1)
	i_to_r_map_s2 = inv_map(r_to_i_map_s2)

	pocket_file_s1 = os.path.join(protein_folder, start + '_pocket.mae')
	pocket_file_s2 = os.path.join(protein_folder, target + '_pocket.mae')
	pocket_s1 = list(structure.StructureReader(pocket_file_s1))[0]
	pocket_s2 = list(structure.StructureReader(pocket_file_s2))[0]
	valid_r_s1 = get_res_near_ligand(r_to_i_map_s1, pocket_s1)
	valid_r_s2 = get_res_near_ligand(r_to_i_map_s2, pocket_s2)

	if valid_r_s1 == set({}):
		print(protein, start, "no residues close to the ligand")
		return

	if valid_r_s1 == 0:
		print(protein, target, "pose viewer file has no ligand")
		return

	if valid_r_s2 == set({}):
		print(protein, start, "no residues close to the ligand")
		return

	if valid_r_s2 == 0:
		print(protein, target, "pose viewer file has no ligand")
		return
	print("Calculating")
	final_r_list_s1 = []
	final_r_list_s2 = []

	for r in valid_r_s1:
		s1index = r_to_i_map_s1[r]

		if paired_str_s1[s1index] == paired_str_s2[s1index]:
			if r not in final_r_list_s1:
				final_r_list_s1.append(r)

			if i_to_r_map_s2[s1index] not in final_r_list_s2:
				final_r_list_s2.append(i_to_r_map_s2[s1index])

	for r in valid_r_s2:
		s2index = r_to_i_map_s2[r]

		if paired_str_s2[s2index] == paired_str_s1[s2index]:
			if r not in final_r_list_s2:
				final_r_list_s2.append(r)

			if i_to_r_map_s1[s2index] not in final_r_list_s1:
				final_r_list_s1.append(i_to_r_map_s1[s2index])

	(asl_list_s1, a_list_s1) = get_atoms(s1, final_r_list_s1)
	(asl_list_s2, a_list_s2) = get_atoms(s2, final_r_list_s2)
	rmsd_ls = []

	for k in range(len(a_list_s1)):
		if len(a_list_s1[k]) == len(a_list_s2[k]):
			rmsd_val = rmsd.calculate_in_place_rmsd(s1, a_list_s1[k], s2, a_list_s2[k])
			rmsd_ls.append(rmsd_val)

	return rmsd_ls


def main():
	task = sys.argv[1]

	if task == 'all':
		for file in os.listdir(align_path):
			if file == 'run':
				continue
			cmd = 'sbatch -p rondror -t 1:00:00 -o {} --wrap="$SCHRODINGER/run python3 rmsd_calculator.py group {}"'
			os.system(cmd.format(os.path.join(run_path, 'rmsd_{}.out'.format(file)), file))

	if task == 'group':
		file = sys.argv[2]
		i = file.strip('alignment.pkl')

		infile = open(os.path.join(align_path, file), 'rb')
		paired_strs = pickle.load(infile)
		infile.close()
		infile = open(chain_path, 'rb')
		chains = pickle.load(infile)
		infile.close()

		rmsds = {}

		for protein in paired_strs:
			rmsds[protein] = {}
			protein_folder = os.path.join(data_root, protein + '/structures/aligned')
			for start in paired_strs[protein]:
				if start not in rmsds[protein]:
					rmsds[protein][start] = {}
				primary_file = os.path.join(protein_folder, start + '_prot.mae')
				if os.path.exists(primary_file):
					s1 = list(structure.StructureReader(primary_file))[0]
				else:
					print("File does not exist:", start + '_prot.mae')
				for target in paired_strs[protein][start]:
					print(protein, start, target)
					primary_file = os.path.join(protein_folder, target + '_prot.mae')
					if os.path.exists(primary_file):
						s2 = list(structure.StructureReader(primary_file))[0]
					else:
						print("File does not exist:", target + '_prot.mae')

					rmsds[protein][start][target] = compute_protein_rmsds(paired_strs, protein, start, target, s1,
																		  s2, chains, protein_folder)
		with open(save_path.format(i), 'wb') as f:
			pickle.dump(rmsds, f)


if __name__ == '__main__':
	main()
