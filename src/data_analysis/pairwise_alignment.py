'''
This protocol can be used to find the pairwise alignemnt between the amino acid strings of each pair of proteins
how to run this file:
/home/groups/rondror/software/sidhikab/miniconda/envs/geometric/bin/python pairwise_alignment.py all
/home/groups/rondror/software/sidhikab/miniconda/envs/geometric/bin/python pairwise_alignment.py group <group index>
'''

from Bio import pairwise2
import os
import sys
import pickle

seq_file = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/sequences.pkl'
run_path = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/alignments/run'
protein_file = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt'
save_path = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/alignments/alignment{}.pkl'

'''
This method finds the pairwise alignemnt between the amino acid strings of each pair of proteins
:param protein: name of the protein
:param seq_file: path to the file containing the amino acid sequence of the protein
:param save_folder: path to the location where the alignment string should be saved
:return:
'''
def compute_protein_alignments(protein, start, target, strs, paired_strs):
	str_s1 = strs[protein][start]
	str_s2 = strs[protein][target]
	alignments = pairwise2.align.globalxx(str_s1, str_s2)
	if start not in paired_strs:
		paired_strs[protein][start] = {}
	paired_strs[protein][start][target] = (alignments[0][0], alignments[0][1])


def main():
	task = sys.argv[1]

	infile = open(seq_file, 'rb')
	strs = pickle.load(infile)
	infile.close()

	proteins = sorted(list(strs.keys()))
	grouped_files = []
	n = 25
	for i in range(0, len(proteins), n):
		grouped_files += [proteins[i: i + n]]

	if task == 'all':
		for i, group in enumerate(grouped_files):
			cmd = 'sbatch -p rondror -t 1:00:00 -o {} --wrap="/home/groups/rondror/software/sidhikab/miniconda/envs/geometric/bin/python pairwise_alignment.py group {}"'
			os.system(cmd.format(os.path.join(run_path, 'align{}.out'.format(i)), i, i))
			# print(cmd.format(os.path.join(run_path, 'align{}.out'.format(i)), i, i))
	if task == 'group':
		i = int(sys.argv[2])
		paired_strs = {}

		with open(protein_file) as fp:
			for line in fp:
				if line[0] == '#': continue
				protein, target, start = line.strip().split()
				if protein in grouped_files[i]:
					paired_strs[protein] = {}
					compute_protein_alignments(protein, start, target, strs, paired_strs)

		with open(save_path.format(i), 'wb') as f:
			pickle.dump(paired_strs, f)


if __name__ == '__main__':
	main()