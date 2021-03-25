"""
The purpose of this code is to first create the raw directory folder and include the following files
starting protein receptor
starting ligand
target ligand
glide pose viewer file

Then the top glide poses are added

Then the decoys are created

It can be run on sherlock using
$ $SCHRODINGER/run python3 model.py /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw
"""

import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    args = parser.parse_args()

    protein = 'O38732'
    target = '2i0a'
    start = '2q5k'

    pair = '{}-to-{}'.format(target, start)
    protein_path = os.path.join(args.raw_root, protein)
    pair_path = os.path.join(protein_path, pair)
    save_folder = os.path.join(pair_path, 'clash_data')

    df = pd.read_csv(os.path.join(save_folder, 'combined.csv'))
    X = df[['bfactor', 'mcss', 'volume_docking']]
    y = df['volume_target']
    y[y <= 20] = 0
    y[y > 20] = 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    print("Accuracy is {}".format(accuracy))
    print("Precision is {}".format(precision))
    print("Recall is {}".format(recall))


if __name__=="__main__":
    main()