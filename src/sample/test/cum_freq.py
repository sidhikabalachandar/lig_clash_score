"""
The purpose of this code is to create the cumulative frequency and bar graphs

It can be run on sherlock using
/home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python cum_freq.py filter_stats /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits/search_test_incorrect_glide_index.txt /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw /home/users/sidhikab/lig_clash_score/reports/figures --group_name exhaustive_grid_1_rotation_5 --protein O38732 --target 2i0a --start 2q5k
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

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
            process.append((protein, target, start))

    return process


def bar_graph(glide_ls, score_no_vdw_ls, pose_ls, out_dir, protein, target, start):
    fig, ax = plt.subplots()
    plt.plot(pose_ls, glide_ls, label='Glide')
    plt.plot(pose_ls, score_no_vdw_ls, label='Score no vdw')

    ax.legend()
    ax.set_xlabel('Pose Cutoff')
    ax.set_ylabel('Min RMSD')
    plt.savefig(os.path.join(out_dir, '{}_{}_{}.png'.format(protein, target, start)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either run, check, or delete_json')
    parser.add_argument('docked_prot_file', type=str, help='file listing proteins to process')
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('out_dir', type=str, help='directory where all graphs will be saved')
    parser.add_argument('--protein', type=str, default='', help='protein name')
    parser.add_argument('--target', type=str, default='', help='target ligand name')
    parser.add_argument('--start', type=str, default='', help='start ligand name')
    parser.add_argument('--group_name', type=str, default='', help='name of pose group subdir')
    parser.add_argument('--max_poses', type=int, default=100, help='maximum number of glide poses considered')
    parser.add_argument('--num_conformers', type=int, default=300, help='maximum number of conformers considered')
    parser.add_argument('--rotation_search_step_size', type=int, default=5, help='step size between each angle '
                                                                                 'checked, in degrees')
    parser.add_argument('--grid_size', type=int, default=1, help='grid size in positive and negative x, y, z '
                                                                 'directions')
    parser.add_argument('--target_clash_cutoff', type=int, default=20, help='clash cutoff between target protein and '
                                                                            'ligand pose')
    parser.add_argument('--intolerable_cutoff', type=int, default=0, help='cutoff of max num intolerable residues')
    args = parser.parse_args()

    random.seed(0)

    if args.task == 'filter_stats':
        if args.protein != '':
            angles = [i for i in range(-30, 30 + args.rotation_search_step_size, args.rotation_search_step_size)]
            grid = []
            for dx in range(-args.grid_size, args.grid_size + 1):
                for dy in range(-args.grid_size, args.grid_size + 1):
                    for dz in range(-args.grid_size, args.grid_size + 1):
                        grid.append([dx, dy, dz])
            num_total = len(grid) * args.num_conformers * len(angles) * len(angles) * len(angles)
            print('Exhaustive search: total = {}'.format(num_total))

            pair = '{}-to-{}'.format(args.target, args.start)
            protein_path = os.path.join(args.raw_root, args.protein)
            pair_path = os.path.join(protein_path, pair)
            pose_path = os.path.join(pair_path, args.group_name)
            clash_path = os.path.join(pose_path, 'clash_data')
            simple_total = 0
            simple_correct = 0
            ideal_total = 0
            ideal_correct = 0
            ideal_res_total = 0
            ideal_res_correct = 0
            pred_total = 0
            pred_correct = 0
            for f in os.listdir(clash_path):
                file = os.path.join(clash_path, f)
                if f[:4] == 'pose':
                    df = pd.read_csv(os.path.join(file))
                    simple_total += len(df)
                    simple_correct += len(df[df['rmsd'] <= 2])

                    ideal_df = df[df['target_clash'] < args.target_clash_cutoff]
                    ideal_total += len(ideal_df)
                    ideal_correct += len(ideal_df[ideal_df['rmsd'] <= 2])

                    ideal_res_df = df[df['true_num_intolerable'] <= args.intolerable_cutoff]
                    ideal_res_total += len(ideal_res_df)
                    ideal_res_correct += len(ideal_res_df[ideal_res_df['rmsd'] <= 2])

                    pred_df = df[df['pred_num_intolerable'] <= args.intolerable_cutoff]
                    pred_total += len(pred_df)
                    pred_correct += len(pred_df[pred_df['rmsd'] <= 2])

            print('Simple starting clash filter: total = {} , correct = {}, proportion = {}'.format(simple_total,
                                                                                                    simple_correct,
                                                                                                    simple_correct / simple_total))
            print('Ideal target clash filter: total = {} , correct = {}, proportion = {}'.format(ideal_total,
                                                                                                 ideal_correct,
                                                                                                 ideal_correct / ideal_total))
            print('Ideal res target clash filter: total = {} , correct = {}, proportion = {}'.format(ideal_res_total,
                                                                                                     ideal_res_correct,
                                                                                                     ideal_res_correct / ideal_res_total))
            print('ML target clash filter: total = {} , correct = {}, proportion = {}'.format(pred_total, pred_correct,
                                                                                              pred_correct / pred_total))
        else:
            pairs = get_prots(args.docked_prot_file)
            random.shuffle(pairs)
            for protein, target, start in pairs[:5]:
                print(protein, target, start)
                angles = [i for i in range(-30, 30 + args.rotation_search_step_size, args.rotation_search_step_size)]
                grid = []
                for dx in range(-args.grid_size, args.grid_size + 1):
                    for dy in range(-args.grid_size, args.grid_size + 1):
                        for dz in range(-args.grid_size, args.grid_size + 1):
                            grid.append([dx, dy, dz])
                num_total = len(grid) * args.num_conformers * len(angles) * len(angles) * len(angles)
                print('Exhaustive search: total = {}'.format(num_total))

                pair = '{}-to-{}'.format(target, start)
                protein_path = os.path.join(args.raw_root, protein)
                pair_path = os.path.join(protein_path, pair)
                pose_path = os.path.join(pair_path, args.group_name)
                clash_path = os.path.join(pose_path, 'clash_data')
                simple_total = 0
                simple_correct = 0
                ideal_total = 0
                ideal_correct = 0
                ideal_res_total = 0
                ideal_res_correct = 0
                pred_total = 0
                pred_correct = 0
                for f in os.listdir(clash_path):
                    file = os.path.join(clash_path, f)
                    if f[:4] == 'pose':
                        df = pd.read_csv(os.path.join(file))
                        simple_total += len(df)
                        simple_correct += len(df[df['rmsd'] <= 2])

                        ideal_df = df[df['target_clash'] < args.target_clash_cutoff]
                        ideal_total += len(ideal_df)
                        ideal_correct += len(ideal_df[ideal_df['rmsd'] <= 2])

                        ideal_res_df = df[df['true_num_intolerable'] <= args.intolerable_cutoff]
                        ideal_res_total += len(ideal_res_df)
                        ideal_res_correct += len(ideal_res_df[ideal_res_df['rmsd'] <= 2])

                        pred_df = df[df['pred_num_intolerable'] <= args.intolerable_cutoff]
                        pred_total += len(pred_df)
                        pred_correct += len(pred_df[pred_df['rmsd'] <= 2])

                print('Simple starting clash filter: total = {} , correct = {}, proportion = {}'.format(simple_total,
                                                                                                        simple_correct,
                                                                                                        simple_correct / simple_total))
                if ideal_total != 0:
                    prop = ideal_correct / ideal_total
                else:
                    prop = 0
                print('Ideal target clash filter: total = {} , correct = {}, proportion = {}'.format(ideal_total,
                                                                                                     ideal_correct,
                                                                                                     prop))
                if ideal_res_total != 0:
                    prop = ideal_res_correct / ideal_res_total
                else:
                    prop = 0
                print(
                    'Ideal res target clash filter: total = {} , correct = {}, proportion = {}'.format(ideal_res_total,
                                                                                                       ideal_res_correct,
                                                                                                       prop))
                if pred_total != 0:
                    prop = pred_correct / pred_total
                else:
                    prop = 0
                print('ML target clash filter: total = {} , correct = {}, proportion = {}'.format(pred_total,
                                                                                                  pred_correct,
                                                                                                  prop))

    elif args.task == 'graph':
        pair = '{}-to-{}'.format(args.target, args.start)
        protein_path = os.path.join(args.raw_root, args.protein)
        pair_path = os.path.join(protein_path, pair)
        df = pd.read_csv(os.path.join(pair_path, 'poses_after_pred_filter.csv'))
        glide_scores = df['glide_score'].tolist()
        score_no_vdws = df['modified_score_no_vdw'].tolist()
        rmsds = df['rmsd'].tolist()
        names = df['name'].tolist()

        glide_df = pd.read_csv(os.path.join(pair_path, '{}.csv'.format(pair)))
        for i in range(1, 100):
            pose_df = glide_df[glide_df['target'] == '{}_lig{}'.format(args.target, i)]
            if len(pose_df) > 0:
                names.append(pose_df['target'].iloc[0])
                rmsds.append(pose_df['rmsd'].iloc[0])
                glide_scores.append(pose_df['glide_score'].iloc[0])
                score = pose_df['score_no_vdw'].iloc[0]
                if score > 20:
                    score_no_vdws.append(20)
                elif score < -20:
                    score_no_vdws.append(-20)
                else:
                    score_no_vdws.append(score)

        glide_data = [(glide_scores[i], rmsds[i], names[i]) for i in range(len(rmsds))]
        score_no_vdw_data = [(score_no_vdws[i], rmsds[i], names[i]) for i in range(len(rmsds))]

        # sort data in reverse rmsd order (make sure that we choose worst in tie breakers)
        rev_glide_data = sorted(glide_data, key=lambda x: x[1], reverse=True)
        rev_score_no_vdw_data = sorted(score_no_vdw_data, key=lambda x: x[1], reverse=True)

        sorted_glide = sorted(rev_glide_data, key=lambda x: x[0])
        sorted_score_no_vdw = sorted(rev_score_no_vdw_data, key=lambda x: x[0])

        glide_ls = []
        score_no_vdw_ls = []
        pose_ls = [i for i in range(1, 100)]

        for i in tqdm(range(1, 100), desc='creating graph'):
            glide_ls.append(min(sorted_glide[:i], key=lambda x: x[1])[1])
            score_no_vdw_ls.append(min(sorted_score_no_vdw[:i], key=lambda x: x[1])[1])
        bar_graph(glide_ls, score_no_vdw_ls, pose_ls, args.out_dir, args.protein, args.target, args.start)


if __name__=="__main__":
    main()