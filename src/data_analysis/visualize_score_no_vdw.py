"""
The purpose of this code is to create the csv files with rmsd, mcss, and physcis score information

the code also combines all of the info for each protein, target, start group into one csv file

It can be run on sherlock using
/home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python visualize_score_no_vdw.py /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/combined.csv /home/users/sidhikab/lig_clash_score/reports/figures/score_no_vdw.png
"""

import argparse
import pandas as pd
import seaborn as sns
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('combined_file', type=str, help='csv file with score_no_vdw data')
    parser.add_argument('save_path', type=str, help='path to saved figure')
    args = parser.parse_args()

    data = pd.read_csv(args.combined_file)
    print(data[data['score_no_vdw'] == max(data['score_no_vdw'])])
    print(data[data['score_no_vdw'] == max(data['score_no_vdw'])]['protein'])
    print(data[data['score_no_vdw'] == max(data['score_no_vdw'])]['start'])
    print(data[data['score_no_vdw'] == max(data['score_no_vdw'])]['target'])
    print(data[data['score_no_vdw'] == min(data['score_no_vdw'])])
    # print(len(data[data['score_no_vdw'] < -50]['start'].unique()) + len(data[data['score_no_vdw'] > 50]['start'].unique()))
    # print(data[data['score_no_vdw'] < -50]['start'].unique())
    data['score_no_vdw'] = [x if x <= 50 else 50 for x in data['score_no_vdw']]
    data['score_no_vdw'] = [x if x >= -50 else -50 for x in data['score_no_vdw']]
    ax = sns.distplot(data['score_no_vdw'])
    fig = ax.get_figure()
    fig.savefig(args.save_path)

if __name__ == "__main__":
    main()