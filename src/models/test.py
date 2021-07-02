"""
The purpose of this code is to train the gnn model

It can be run on sherlock using
$ sbatch 1gpu.sbatch /home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python test_subsample_incorrect.py /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d
"""
f = open("/home/users/sidhikab/lig_clash_score/src/models/out/test.out", "w")
f.write("start import\n")
f.close()
import os
import argparse
from pdbbind_dataloader import pdbbind_dataloader
f = open("/home/users/sidhikab/lig_clash_score/src/models/out/test.out", "a")
f.write('end import\n')
f.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=str, help='directory where raw and processed directories can be found')
    parser.add_argument('--split', type=str, default='random', help='name of split files')
    args = parser.parse_args()

    batch_size = 700
    data_path = args.root
    split_dir = os.path.join(data_path, 'splits')
    train_split = os.path.join(split_dir, f'train_{args.split}.txt')
    f = open("/home/users/sidhikab/lig_clash_score/src/models/out/test.out", "a")
    f.write('getting train_loader\n')
    f.close()
    pdbbind_dataloader(batch_size, data_dir=data_path, split_file=train_split)
    f = open("/home/users/sidhikab/lig_clash_score/src/models/out/test.out", "a")
    f.write('done')
    f.close()

if __name__=="__main__":
    main()