"""
The purpose of this code is to get the physics scores and the rmsds

It can be run on sherlock using
$ $SCHRODINGER/run python3 check_labels.py
"""

import pandas as pd


def main():
    df = pd.read_csv('/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/ml_score/data/names.csv')
    print(df.label.unique())


if __name__=="__main__":
    main()