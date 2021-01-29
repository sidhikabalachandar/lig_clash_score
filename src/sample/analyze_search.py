"""
The purpose of this code is to create conformers

It can be run on sherlock using
$ $SCHRODINGER/run python3 analyze_search.py analyze_clash_data /home/users/sidhikab/lig_clash_score/src/data/decoy_timing_data_with_clash --grid_file  /home/users/sidhikab/lig_clash_score/src/data/decoy_timing_data/grid_data.pkl --grid_size 1 --n 1
"""

import argparse
import os
import random
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# HELPER FUNCTIONS


def group_grid(n, grid_size):
    grid = []
    for dx in range(-grid_size, grid_size + 1):
        for dy in range(-grid_size, grid_size + 1):
            for dz in range(-grid_size, grid_size + 1):
                grid.append([dx, dy, dz])

    grouped_files = []

    for i in range(0, len(grid), n):
        grouped_files += [grid[i: i + n]]

    return grouped_files


def get_grid_prots(grid_file, grid_size):
    infile = open(grid_file, 'rb')
    data = pickle.load(infile)
    infile.close()

    grid_prots = []
    for pair in data:
        if sum(data[pair][:grid_size + 1]) != 0:
            grid_prots.append(pair)

    return grid_prots


def clash_data(pairs, grouped_files, save_folder, rmsd_cutoff, clash_cutoff):
    # for protein, target, start in tqdm(pairs, desc="going through protein-ligand pairs to get clash data"):
    #     pair_folder = os.path.join(save_folder, '{}_{}-to-{}'.format(protein, target, start))
    #     for i in range(len(grouped_files)):
    #         infile = open(os.path.join(pair_folder, 'clash_data_{}.pkl'.format(i)), 'rb')
    #         data = pickle.load(infile)
    #         infile.close()
    #
    #         num_after_clash_filter = len([x for x in data if x[0] < clash_cutoff])
    #         num_correct_after_clash_filter = len([x for x in data if x[0] < clash_cutoff and x[1] < rmsd_cutoff])
    #
    #         data_file = os.path.join(pair_folder, '{}.csv'.format(i))
    #         df = pd.read_csv(data_file)
    #         df['num_after_clash_filter_cutoff_{}'.format(clash_cutoff)] = [num_after_clash_filter]
    #         df['num_correct_after_clash_filter_cutoff_{}'.format(clash_cutoff)] = [num_correct_after_clash_filter]
    #         df.to_csv(data_file)

    for protein, target, start in tqdm(pairs, desc="going through protein-ligand pairs to get clash data"):
        print(protein, target, start)
        start_folder = os.path.join(os.getcwd(), 'decoy_timing_data_with_clash', '{}_{}-to-{}'.format(protein, target, start))
        target_folder = os.path.join(os.getcwd(), 'decoy_timing_data_with_target_clash', '{}_{}-to-{}'.format(protein, target,
                                                                                                 start))
        for i in range(len(grouped_files)):
            infile = open(os.path.join(start_folder, 'clash_data_{}.pkl'.format(i)), 'rb')
            start_clash_data = pickle.load(infile)
            infile.close()

            infile = open(os.path.join(target_folder, 'clash_data_{}.pkl'.format(i)), 'rb')
            target_clash_data = pickle.load(infile)
            infile.close()

            data_file = os.path.join(start_folder, '{}.csv'.format(i))
            df = pd.read_csv(data_file)
            df = df.rename(columns={'num_after_clash_filter_cutoff_200': 'num_after_simple_200',
                                        'num_correct_after_clash_filter_cutoff_200': 'num_correct_after_simple_200',
                                        'num_after_clash_filter_cutoff_100': 'num_after_simple_100',
                                        'num_correct_after_clash_filter_cutoff_100': 'num_correct_after_simple_100',
                                        'num_after_target_clash_filter_cutoff_15': 'num_after_ideal_advanced_15',
                                        'num_correct_after_target_clash_filter_cutoff_15':
                                            'num_correct_after_ideal_advanced_15'})

            num_after_clash_filter = len([x for x in start_clash_data if x[0] < 200])
            num_correct_after_clash_filter = len([x for x in start_clash_data if x[0] < 200 and x[1] < rmsd_cutoff])
            df['num_after_simple_200'.format(clash_cutoff)] = [num_after_clash_filter]
            df['num_correct_after_simple_200'.format(clash_cutoff)] = [num_correct_after_clash_filter]

            num_after_clash_filter = len([x for x in start_clash_data if x[0] < 100])
            num_correct_after_clash_filter = len([x for x in start_clash_data if x[0] < 100 and x[1] < rmsd_cutoff])
            df['num_after_simple_100'.format(clash_cutoff)] = [num_after_clash_filter]
            df['num_correct_after_simple_100'.format(clash_cutoff)] = [num_correct_after_clash_filter]

            num_after_clash_filter = len([x for x in target_clash_data if x[0] < 15])
            num_correct_after_clash_filter = len([x for x in target_clash_data if x[0] < 15 and x[1] < rmsd_cutoff])
            df['num_after_ideal_advanced_15'.format(clash_cutoff)] = [num_after_clash_filter]
            df['num_correct_after_ideal_advanced_15'.format(clash_cutoff)] = [num_correct_after_clash_filter]

            num_after_clash_filter = 0
            num_correct_after_clash_filter = 0

            for clash_index in range(len(target_clash_data)):
                if target_clash_data[clash_index][0] < 15 and start_clash_data[clash_index][0] < 100:
                    num_after_clash_filter += 1
                    if target_clash_data[clash_index][1] < rmsd_cutoff:
                        num_correct_after_clash_filter += 1

            df['num_after_simple_100_ideal_advanced_15'.format(clash_cutoff)] = [num_after_clash_filter]
            df['num_correct_after_simple_100_ideal_advanced_15'.format(clash_cutoff)] = [num_correct_after_clash_filter]

            df.to_csv(data_file)
            print(data_file)


def analyze_clash_data(pairs, save_folder):
    df = pd.read_csv(os.path.join(save_folder, 'combined_with_stacked_filter.csv'))
    for protein, target, start in pairs:
        print(protein, target, start)
        for cutoff in [100, 200]:
            print('\tCUTOFF =', cutoff)
            pair_df = df[(df['protein'] == protein) & (df['target'] == target) & (df['start'] == start)]
            fraction_before_filter = sum(pair_df['num_correct_poses_found']) / sum(pair_df['num_poses_searched'])
            fraction_after_filter = sum(pair_df['num_correct_after_simple_{}'.format(cutoff)]) / \
                                    sum(pair_df['num_after_simple_{}'.format(cutoff)])

            print('\t\tfraction_before_filter =', fraction_before_filter, ', fraction_after_filter =',
                  fraction_after_filter)

            num_poses_at_start = sum(pair_df['num_poses_searched'])
            num_poses_after_clash_filter = sum(pair_df['num_after_simple_{}'.format(cutoff)])
            num_cut = num_poses_at_start - num_poses_after_clash_filter
            print('\t\tnum_poses_at_start =', num_poses_at_start, ', num_poses_after_filter =',
                  num_poses_after_clash_filter, ', num_poses_cut =', num_cut, ', fraction_cut =',
                  num_cut / num_poses_at_start)

            num_correct_at_start = sum(pair_df['num_correct_poses_found'])
            num_correct_after_clash_filter = sum(pair_df['num_correct_after_simple_{}'.format(cutoff)])
            num_cut = num_correct_at_start - num_correct_after_clash_filter
            print('\t\tnum_correct_at_start =', num_correct_at_start, ', num_correct_after_filter =',
                  num_correct_after_clash_filter, ', num_poses_cut =', num_cut, ', fraction_cut =',
                  num_cut / num_correct_at_start)

        cutoff = 15
        print('\tTARGET CLASH CUTOFF =', cutoff)
        pair_df = df[(df['protein'] == protein) & (df['target'] == target) & (df['start'] == start)]
        fraction_before_filter = sum(pair_df['num_correct_poses_found']) / sum(pair_df['num_poses_searched'])
        fraction_after_filter = sum(pair_df['num_correct_after_ideal_advanced_{}'.format(cutoff)]) / \
                                sum(pair_df['num_after_ideal_advanced_{}'.format(cutoff)])

        print('\t\tfraction_before_filter =', fraction_before_filter, ', fraction_after_filter =',
              fraction_after_filter)

        num_poses_at_start = sum(pair_df['num_poses_searched'])
        num_poses_after_clash_filter = sum(pair_df['num_after_ideal_advanced_{}'.format(cutoff)])
        num_cut = num_poses_at_start - num_poses_after_clash_filter
        print('\t\tnum_poses_at_start =', num_poses_at_start, ', num_poses_after_filter =',
              num_poses_after_clash_filter,
              ', num_poses_cut =', num_cut, ', fraction_cut =', num_cut / num_poses_at_start)

        num_correct_at_start = sum(pair_df['num_correct_poses_found'])
        num_correct_after_clash_filter = sum(pair_df['num_correct_after_ideal_advanced_{}'.format(cutoff)])
        num_cut = num_correct_at_start - num_correct_after_clash_filter
        print('\t\tnum_correct_at_start =', num_correct_at_start, ', num_correct_after_filter =',
              num_correct_after_clash_filter, ', num_poses_cut =', num_cut, ', fraction_cut =',
              num_cut / num_correct_at_start)

        print('\tSTART CLASH CUTOFF = 100 TARGET CLASH CUTOFF = 15')
        pair_df = df[(df['protein'] == protein) & (df['target'] == target) & (df['start'] == start)]
        fraction_before_filter = sum(pair_df['num_correct_poses_found']) / sum(pair_df['num_poses_searched'])
        fraction_after_filter = sum(pair_df['num_correct_after_simple_100_ideal_advanced_15']) / \
                                sum(pair_df['num_after_simple_100_ideal_advanced_15'])

        print('\t\tfraction_before_filter =', fraction_before_filter, ', fraction_after_filter =',
              fraction_after_filter)

        num_poses_at_start = sum(pair_df['num_poses_searched'])
        num_poses_after_clash_filter = sum(pair_df['num_after_simple_100_ideal_advanced_15'])
        num_cut = num_poses_at_start - num_poses_after_clash_filter
        print('\t\tnum_poses_at_start =', num_poses_at_start, ', num_poses_after_filter =',
              num_poses_after_clash_filter,
              ', num_poses_cut =', num_cut, ', fraction_cut =', num_cut / num_poses_at_start)

        num_correct_at_start = sum(pair_df['num_correct_poses_found'])
        num_correct_after_clash_filter = sum(pair_df['num_correct_after_simple_100_ideal_advanced_15'])
        num_cut = num_correct_at_start - num_correct_after_clash_filter
        print('\t\tnum_correct_at_start =', num_correct_at_start, ', num_correct_after_filter =',
              num_correct_after_clash_filter, ', num_poses_cut =', num_cut, ', fraction_cut =',
              num_cut / num_correct_at_start)


def graph_clash(pairs, grouped_files, save_folder, rmsd_cutoff):
    data = []
    for protein, target, start in tqdm(pairs, desc="going through protein-ligand pairs to get clash data"):
        pair_folder = os.path.join(save_folder, '{}_{}-to-{}'.format(protein, target, start))
        for i in range(len(grouped_files)):
            infile = open(os.path.join(pair_folder, 'clash_data_{}.pkl'.format(i)), 'rb')
            data = pickle.load(infile)
            infile.close()
            data.extend(data)

    correct = [x[0] for x in data if x[1] < rmsd_cutoff and x[0] < 200]
    incorrect = [x[0] for x in data if x[1] >= rmsd_cutoff]
    random.shuffle(incorrect)
    incorrect = incorrect[:2000000]
    incorrect = [x for x in incorrect if x < 200]

    fig, ax = plt.subplots()
    sns.distplot(correct, hist=True, label='RMSD < 2 A')
    sns.distplot(incorrect, hist=True, label='RMSD >= 2 A')
    ax.legend()
    ax.set_xlabel('Clash Volume')
    ax.set_ylabel('Frequency')
    plt.savefig(os.path.join(save_folder, 'target_clash_frequency_sub_200.png'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either align or search')
    parser.add_argument('save_folder', type=str, help='directory where data will be saved')
    parser.add_argument('--n', type=int, default=30, help='number of grid_points processed in each job')
    parser.add_argument('--grid_size', type=int, default=6, help='grid size in positive and negative x, y, z '
                                                                 'directions')
    parser.add_argument('--grid_file', type=str, default='', help='pickle file with grid data dictionary')
    parser.add_argument('--rmsd_cutoff', type=int, default=2, help='rmsd accuracy cutoff between predicted ligand pose '
                                                                   'and true ligand pose')
    parser.add_argument('--clash_cutoff', type=int, default=100, help='clash volume cutoff for filter')
    args = parser.parse_args()

    random.seed(0)

    if args.task == 'clash_data':
        grouped_files = group_grid(args.n, args.grid_size)
        pairs = get_grid_prots(args.grid_file, args.grid_size)
        clash_data(pairs, grouped_files, args.save_folder, args.rmsd_cutoff, args.clash_cutoff)

    elif args.task == 'analyze_clash_data':
        pairs = get_grid_prots(args.grid_file, args.grid_size)
        analyze_clash_data(pairs, args.save_folder)

    elif args.task == 'graph_clash':
        grouped_files = group_grid(args.n, args.grid_size)
        pairs = get_grid_prots(args.grid_file, args.grid_size)
        graph_clash(pairs, grouped_files, args.save_folder, args.rmsd_cutoff)


if __name__ == "__main__":
    main()
