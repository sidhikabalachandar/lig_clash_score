"""
The purpose of this code is to create conformers

It can be run on sherlock using
$ $SCHRODINGER/run python3 group_poses.py check /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/raw --protein P18031 --target 1g7g --start 1c83
"""

import argparse
import os
import schrodinger.structure as structure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='either align or search')
    parser.add_argument('raw_root', type=str, help='directory where raw data will be placed')
    parser.add_argument('--protein', type=str, default='', help='protein name')
    parser.add_argument('--target', type=str, default='', help='target ligand name')
    parser.add_argument('--start', type=str, default='', help='start ligand name')
    parser.add_argument('--num_poses_per_file', type=int, default=300, help='number of poses in each file')
    args = parser.parse_args()

    if args.task == 'run':
        pair = '{}-to-{}'.format(args.target, args.start)
        protein_path = os.path.join(args.raw_root, args.protein)
        pair_path = os.path.join(protein_path, pair)
        pose_path = os.path.join(pair_path, 'grid_search_poses')
        grouped_pose_path = os.path.join(pose_path, 'grouped_poses')
        counter = 0

        if not os.path.exists(grouped_pose_path):
            os.mkdir(grouped_pose_path)

        for file in os.listdir(pose_path):
            if file[-5:] == 'maegz':
                grid_pt_poses = list(structure.StructureReader(os.path.join(pose_path, file)))
                num_grid_poses = len(grid_pt_poses)
                for i in range(num_grid_poses // args.num_poses_per_file + 1):
                    group_file = os.path.join(grouped_pose_path, '{}.maegz'.format(counter))
                    counter += 1
                    with structure.StructureWriter(group_file) as group_poses:
                        for j in range(args.num_poses_per_file):
                            num = i * args.num_poses_per_file + j
                            if num >= num_grid_poses:
                                break
                            else:
                                pose = grid_pt_poses[num]
                                conf, gridloc, rot = pose.title.split('_')
                                conf = int(conf[4:])
                                gridloc_x, gridloc_y, gridloc_z = gridloc[7:].split(',')
                                gridloc_x = int(gridloc_x)
                                gridloc_y = int(gridloc_y)
                                gridloc_z = int(gridloc_z)
                                rot_x, rot_y, rot_z = rot[3:].split(',')
                                rot_x = int(rot_x)
                                rot_y = int(rot_y)
                                rot_z = int(rot_z)
                                pose.title = '{}_{},{},{}_{},{},{}'.format(conf, gridloc_x, gridloc_y, gridloc_z, rot_x,
                                                                           rot_y,
                                                                           rot_z)
                                group_poses.append(pose)

    elif args.task == 'check':
        pair = '{}-to-{}'.format(args.target, args.start)
        protein_path = os.path.join(args.raw_root, args.protein)
        pair_path = os.path.join(protein_path, pair)
        pose_path = os.path.join(pair_path, 'grid_search_poses')
        grouped_pose_path = os.path.join(pose_path, 'grouped_poses')
        num_grouped_poses = 0

        for file in os.listdir(grouped_pose_path):
            group = list(structure.StructureReader(os.path.join(grouped_pose_path, file)))
            num_grouped_poses += len(group)

        correct_num_poses = 0
        for file in os.listdir(pose_path):
            if file[-5:] == 'maegz':
                grid_pt_poses = list(structure.StructureReader(os.path.join(pose_path, file)))
                num_grid_poses = len(grid_pt_poses)
                correct_num_poses += num_grid_poses

        if num_grouped_poses != correct_num_poses:
            print('mismatch in num poses: grouped {} poses, should have grouped {} poses'.format(num_grouped_poses,
                                                                                                 correct_num_poses))
        else:
            print("all poses grouped")


if __name__ == "__main__":
    main()
