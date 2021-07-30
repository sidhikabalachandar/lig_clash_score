import scipy.spatial
from Bio import pairwise2
import statistics
import schrodinger.structure as structure
import os
from schrodinger.structutils.transform import get_centroid
import numpy as np

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


def bfactor_stats(s):
    '''
    This function gets the mean and standard deviation of all of the bfactors of a structure
    :param s: the protein structure
    :return: the mean and the standard deviation of the list of bfactors associated with the protein structure
    '''
    bfactors = []
    for m in list(s.molecule):
        for r in list(m.residue):
            bfactors.append(r.temperature_factor)
    return statistics.mean(bfactors), statistics.stdev(bfactors)


def normalizedBFactor(r, avg, sdev):
    '''
    This function finds the normalized bfactor for a particular residue
    :param residues: a list of all residues in the protein structure
    :param index: the index of the particular residue in question
    :param avg: the average bfactor over all residues in the protein structure
    :param sdev: the standard deviation calculated over all residues in the protein structure
    :return: the normalized bfactor value
    '''
    return (r.temperature_factor - avg) / sdev


def get_sequence_from_str(s):
    '''
    Get the amino acid sequence
    :param file: .mae file for the structure
    :return: the amino acid string for all amino acids in chain A
    '''
    str = ''
    for m in list(s.molecule):
        for r in list(m.residue):
            str += r.getCode()
    return str


def compute_protein_alignments(seq_docking, seq_target):
    '''
    This method finds the pairwise alignemnt between the amino acid strings of each pair of proteins
    :param protein: name of the protein
    :param seq_file: path to the file containing the amino acid sequence of the protein
    :param save_folder: path to the location where the alignment string should be saved
    :return:
    '''
    alignments = pairwise2.align.globalxx(seq_docking, seq_target)
    return alignments[0][0], alignments[0][1]


def get_all_res_asl(s):
    '''
    This function gets the pdbcode, chain, resnum, and getCode() of every residue in the protein structure
    It ignores any residues associated with the ligand
    :param s: the protein structure
    :return: the list of every residue's pdbcode, chain, resnum, and getCode()
    '''
    r_list = []
    for m in list(s.molecule):
        for r in list(m.residue):
            r_list.append((r.getCode(), r.getAsl()))
    return r_list


def get_all_res_atoms(s):
    '''
    This function gets the pdbcode, chain, resnum, and getCode() of every residue in the protein structure
    It ignores any residues associated with the ligand
    :param s: the protein structure
    :return: the list of every residue's pdbcode, chain, resnum, and getCode()
    '''
    r_list = []
    for m in list(s.molecule):
        for r in list(m.residue):
            r_list.append((r.getCode(), tuple(r.getAtomIndices())))
    return r_list


def map_residues_to_align_index(alignment_string, r_list):
    '''
    Maps unique residue identifiers to list index in alignment string
    :param alignment_string: (string) output from alignment program, contains one letter codes and dashes
    	example: 'TE--S--T-'
    :param r_list: list of unique identifiers of each residue in order of sequence
    	number of residues in r_list must be equal to number of residues in alignment_string
    :return: the map of residues to alignment_string index
    '''
    r_to_i_map = {}
    counter = 0
    for i in range(len(alignment_string)):
        if counter >= len(r_list):
            break
        if alignment_string[i] == r_list[counter][0]:
            r_to_i_map[r_list[counter]] = i
            counter += 1
    return r_to_i_map


def map_index_to_residue(alignment_string, r_list):
    '''
    Maps unique residue identifiers to list index in alignment string
    :param alignment_string: (string) output from alignment program, contains one letter codes and dashes
    	example: 'TE--S--T-'
    :param r_list: list of unique identifiers of each residue in order of sequence
    	number of residues in r_list must be equal to number of residues in alignment_string
    :return: the map of residues to alignment_string index
    '''
    i_to_r_map = {}
    counter = 0
    for i in range(len(alignment_string)):
        if counter >= len(r_list):
            break
        if alignment_string[i] == r_list[counter][0]:
            i_to_r_map[i] = r_list[counter]
            counter += 1
    return i_to_r_map


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

    grid_size = max(grid_size, 6)
    return grid_size


def get_res(s):
    res = []

    for m in list(s.molecule):
        for r in list(m.residue):
            res.append(r)

    return res
