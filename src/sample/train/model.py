"""
The purpose of this code is to first create the raw directory folder and include the following files
starting protein receptor
starting ligand
target ligand
glide pose viewer file

Then the top glide poses are added

Then the decoys are created

It can be run on sherlock using
$ $SCHRODINGER/run python3 model.py train /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d --save_path /home/users/sidhikab/lig_clash_score/reports/figures/bfactor_vs_volume.png --docked_prot_file /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/refined_random.txt
"""

import argparse
import os
import pandas as pd
from sklearn import tree
from sklearn import metrics
import random
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt


def get_prots(docked_prot_file):
    """
    gets list of all protein, target ligands, and starting ligands in the index file
    :param docked_prot_file: (string) file listing proteins to process
    :return: process (list) list of all protein, target ligands, and starting ligands to process
    """
    process = []
    with open(docked_prot_file) as fp:
        for line in fp:
            if line[0] == '#':
                continue
            protein, target, start = line.strip().split()
            if protein not in process:
                process.append(protein)

    return process


def train_test_split(df, root):
    proteins = df.protein.unique()
    random.shuffle(proteins)
    train_prots = get_prots(os.path.join(root, 'splits', 'search_train_index.txt'))
    test_prots = get_prots(os.path.join(root, 'splits', 'search_test_index.txt'))
    print(len(train_prots), len(test_prots))
    test_dfs = []
    train_dfs = []
    for prot in proteins:
        prot_df = df[df['protein'] == prot]
        if prot in train_prots:
            train_dfs.append(prot_df)
        elif prot in test_prots:
            test_dfs.append(prot_df)
        else:
            print(prot)
    test = pd.concat(test_dfs)
    train = pd.concat(train_dfs)

    print('num test proteins:', len(test.protein.unique()))
    X_test = test[['bfactor', 'mcss', 'volume_docking']]
    y_test = test['volume_target']
    y_test[y_test <= 20] = 0
    y_test[y_test > 20] = 1
    X_train = train[['bfactor', 'mcss', 'volume_docking']]
    y_train = train['volume_target']
    y_train[y_train <= 20] = 0
    y_train[y_train > 20] = 1
    print('num intolerable clash: {}, total: {}, proportion intolerable clash: {}'.format(sum(y_test) + sum(y_test),
          len(y_train) + len(y_test), (sum(y_test) + sum(y_test)) / (len(y_train) + len(y_test))))
    print('test size is {} of total size'.format(len(y_test) / len(df)))
    return X_train, X_test, y_train, y_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either align or search')
    parser.add_argument('root', type=str, help='directory where raw data will be placed')
    parser.add_argument('--save_path', type=str, default="", help='grid point group index')
    parser.add_argument('--docked_prot_file', type=str, help='file listing proteins to process')
    args = parser.parse_args()
    random.seed(0)

    raw_root = os.path.join(args.root, 'raw')

    if args.task == 'train':
        df = pd.read_csv(os.path.join(args.root, 'combined_clash_data.csv'))
        X_train, X_test, y_train, y_test = train_test_split(df, args.root)
        clf_file = os.path.join(args.root, 'clash_classifier.pkl')
        if os.path.exists(clf_file):
            infile = open(clf_file, 'rb')
            clf = pickle.load(infile)
            infile.close()
        else:
            clf = tree.DecisionTreeClassifier()
            clf = clf.fit(X_train, y_train)
            outfile = open(clf_file, 'wb')
            pickle.dump(clf, outfile)
        print('TRAIN')
        y_train_pred = clf.predict(X_train)
        accuracy = metrics.accuracy_score(y_train, y_train_pred)
        precision = metrics.precision_score(y_train, y_train_pred)
        recall = metrics.recall_score(y_train, y_train_pred)
        print("Accuracy is {}".format(accuracy))
        print("Precision is {}".format(precision))
        print("Recall is {}".format(recall))
        print('TEST')
        y_test_pred = clf.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_test_pred)
        precision = metrics.precision_score(y_test, y_test_pred)
        recall = metrics.recall_score(y_test, y_test_pred)
        print("Accuracy is {}".format(accuracy))
        print("Precision is {}".format(precision))
        print("Recall is {}".format(recall))
        importances = clf.feature_importances_
        print("Feature importances: ", importances)

    elif args.task == 'graph':
        fig, ax = plt.subplots()
        df = pd.read_csv(os.path.join(args.root, 'combined_clash_data.csv'))
        proteins = df.protein.unique()
        all_tolerable_bfactors = []
        all_intolerable_bfactors = []
        all_tolerable_volume_dockings = []
        all_intolerable_volume_dockings = []
        for prot in tqdm(proteins, desc='creating data sets'):
            prot_df = df[df['protein'] == prot]
            tolerable_df = prot_df[prot_df['volume_target'] <= 20]
            intolerable_df = prot_df[prot_df['volume_target'] > 20]
            tolerable_bfactors = tolerable_df['bfactor'].to_list()
            tolerable_volume_dockings = tolerable_df['volume_docking'].to_list()
            intolerable_bfactors = intolerable_df['bfactor'].to_list()
            intolerable_volume_dockings = intolerable_df['volume_docking'].to_list()
            all_tolerable_bfactors.extend(tolerable_bfactors)
            all_intolerable_bfactors.extend(intolerable_bfactors)
            all_tolerable_volume_dockings.extend(tolerable_volume_dockings)
            all_intolerable_volume_dockings.extend(intolerable_volume_dockings)

        plot_tolerable_bfactors = []
        plot_intolerable_bfactors = []
        plot_tolerable_volume_dockings = []
        plot_intolerable_volume_dockings = []
        for i in range(len(all_intolerable_bfactors)):
            if i % 100 == 0:
                plot_intolerable_bfactors.append(all_intolerable_bfactors[i])
                plot_intolerable_volume_dockings.append(all_intolerable_volume_dockings[i])

        for i in range(len(all_tolerable_bfactors)):
            if i % 100 == 0:
                plot_tolerable_bfactors.append(all_tolerable_bfactors[i])
                plot_tolerable_volume_dockings.append(all_tolerable_volume_dockings[i])
        plt.scatter(plot_intolerable_bfactors, plot_intolerable_volume_dockings, alpha=0.1, label='intolerable')
        plt.scatter(plot_tolerable_bfactors, plot_tolerable_volume_dockings, alpha=0.1, label='tolerable')
        plt.title('bfactor vs volume docking for decision tree model')
        plt.xlabel('bfactor')
        plt.ylabel('volume_dockings')
        ax.legend()
        fig.savefig(args.save_path)

    elif args.task == 'save_split':
        pairs = get_prots(args.docked_prot_file)
        df = pd.read_csv(os.path.join(args.root, 'combined_clash_data.csv'))
        proteins = df.protein.unique()
        random.shuffle(proteins)
        test_prots = []
        train_prots = []
        cur_test_size = 0
        for prot in tqdm(proteins, desc='creating data sets'):
            prot_df = df[df['protein'] == prot]
            if cur_test_size / len(df) < args.test_size or prot == 'O38732':
                cur_test_size += len(prot_df)
                test_prots.append(prot)
            else:
                train_prots.append(prot)

        test_pairs = []
        for p in pairs:
            if p[0] in test_prots:
                test_pairs.append('{} {} {}\n'.format(p[0], p[1], p[2]))


        test_file = open(os.path.join(args.root, 'splits', 'search_test_index.txt'), "w")
        test_file.writelines(test_pairs)
        test_file.close()

        train_pairs = []
        for p in pairs:
            if p[0] in train_prots:
                train_pairs.append('{} {} {}\n'.format(p[0], p[1], p[2]))

        train_file = open(os.path.join(args.root, 'splits', 'search_train_index.txt'), "w")
        train_file.writelines(train_pairs)
        train_file.close()

    elif args.task == 'find_prots':
        pairs_to_check = []
        total_num_prots = 0
        with open(os.path.join(args.root, 'splits', 'search_test_index.txt')) as fp:
            for line in tqdm(fp, desc='protein-ligand pairs'):
                if line[0] == '#':
                    continue
                total_num_prots += 1
                protein, target, start = line.strip().split()
                pair = '{}-to-{}'.format(target, start)
                protein_path = os.path.join(raw_root, protein)
                pair_path = os.path.join(protein_path, pair)
                file = os.path.join(pair_path, '{}.csv'.format(pair))
                df = pd.read_csv(file)
                add = True
                for i in range(1, 100):
                    lig_df = df[df['target'] == '{}_lig{}'.format(target, i)]
                    if len(lig_df) > 0 and lig_df['rmsd'].iloc[0] < 2:
                        add = False
                        break

                if add:
                    pairs_to_check.append('{} {} {}\n'.format(protein,target, start))

        print(len(pairs_to_check), total_num_prots)
        train_file = open(os.path.join(args.root, 'splits', 'search_test_incorrect_glide_index.txt'), "w")
        train_file.writelines(pairs_to_check)
        train_file.close()


if __name__=="__main__":
    main()