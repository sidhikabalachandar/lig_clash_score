'''
This protocol can be used to find the rmsd between the residues in the binding pocket of every pair of structures of a protein
Only the residues within 4 angstroms of either structures' ligands are considered

# how to run this file:
# /home/groups/rondror/software/sidhikab/miniconda/envs/geometric/bin/python flex_res.py
'''

import os
import pickle
import statistics
import matplotlib.pyplot as plt

run_path = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/rmsd/run'
data_root = '/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/rmsd'
cutoff = 2

def main():
    rmsds = {}

    for file in os.listdir(data_root):
        if file == 'run':
            continue
        infile = open(os.path.join(data_root, file), 'rb')
        rmsds_add = pickle.load(infile)
        infile.close()
        rmsds.update(rmsds_add)

    res_data = {}
    error_count = 0
    for protein in rmsds:
        for start in rmsds[protein]:
            for target in rmsds[protein][start]:
                if rmsds[protein][start][target] == None:
                    error_count += 1
                else:
                    num_res = len(rmsds[protein][start][target])
                    num_flex_res = len([i for i in rmsds[protein][start][target] if i > cutoff])
                    if protein not in res_data:
                        res_data[protein] = {}
                    if start not in res_data[protein]:
                        res_data[protein][start] = {}
                    res_data[protein][start][target] = num_res, num_flex_res
    print(error_count, 'docking runs had errors')

    graph_data = {}
    res_ls = []
    for protein in res_data:
        protein_flex_res_ls = []
        for start in res_data[protein]:
            for target in res_data[protein][start]:
                res_ls.append(res_data[protein][start][target][0])
                protein_flex_res_ls.append(res_data[protein][start][target][1])

        graph_data[protein] = statistics.mean(protein_flex_res_ls)

    print("Total Average Number of Residues in Binding Pocket:", statistics.mean(res_ls))
    fig, ax = plt.subplots()
    sorted_graph_data = sorted(graph_data.items(), key=lambda x: x[1], reverse=True)
    ax.barh([i[0] for i in sorted_graph_data], [i[1] for i in sorted_graph_data])
    ax.set(xlabel='Average Number of Flexible Residues', ylabel='Protein Name',
           title='Average Number of Flexible Residues per Protein')
    ax.tick_params(axis='y', labelsize=8)
    n = 20  # Keeps every 7th label
    [l.set_visible(False) for (i, l) in enumerate(ax.yaxis.get_ticklabels()) if i % n != 0]
    plt.savefig('/home/users/sidhikab/flexibility_project/atom3d/reports/figures/flex_res.png')



if __name__ == '__main__':
	main()