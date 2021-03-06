"""
The purpose of this code is to train the gnn model

It can be run on sherlock using
$ sbatch 1gpu_gnn_score_feat.sbatch /home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python train_pdbbind_score_layer.py /home/users/sidhikab/lig_clash_score/models /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/processed_clustered_score /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/splits --mode train_test --split balance_clash --log_dir gnn_score_clustered
$ sbatch 1gpu.sbatch /home/groups/rondror/software/sidhikab/miniconda/envs/test_env/bin/python train_pdbbind_score_layer.py /home/users/sidhikab/lig_clash_score/models /oak/stanford/groups/rondror/projects/combind/flexibility/atom3d --mode test --log_dir v1
"""
import os
import time
import logging
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime
import argparse
import pickle

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, MSELoss
from torch_geometric.nn import GCNConv, GINConv, global_add_pool

from pdbbind_dataloader import pdbbind_dataloader

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim*2)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim*2)
        self.conv3 = GCNConv(hidden_dim*2, hidden_dim*4)
        self.bn3 = torch.nn.BatchNorm1d(hidden_dim*4)
        self.conv4 = GCNConv(hidden_dim*4, hidden_dim*4)
        self.bn4 = torch.nn.BatchNorm1d(hidden_dim*4)
        self.conv5 = GCNConv(hidden_dim*4, hidden_dim*8)
        self.bn5 = torch.nn.BatchNorm1d(hidden_dim*8)
        self.fc1 = Linear(hidden_dim*8, hidden_dim*4)
        self.fc2 = Linear(hidden_dim*4, 1)


    def forward(self, x, edge_index, edge_weight, batch, physics_score):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.conv3(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.bn3(x)
        x = self.conv4(x, edge_index, edge_weight)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv5(x, edge_index, edge_weight)
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.25, training=self.training)
        return self.fc2(x).view(-1)

class GCN_score_layer(torch.nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(GCN_score_layer, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim*2)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim*2)
        self.conv3 = GCNConv(hidden_dim*2, hidden_dim*4)
        self.bn3 = torch.nn.BatchNorm1d(hidden_dim*4)
        self.conv4 = GCNConv(hidden_dim*4, hidden_dim*4)
        self.bn4 = torch.nn.BatchNorm1d(hidden_dim*4)
        self.conv5 = GCNConv(hidden_dim*4, hidden_dim*8)
        self.bn5 = torch.nn.BatchNorm1d(hidden_dim*8)
        self.fc1 = Linear(hidden_dim*8, hidden_dim*4)
        self.fc2 = Linear(hidden_dim*4, 1)
        self.hybrid_score = Linear(2, 1)


    def forward(self, x, edge_index, edge_weight, batch, physics_score):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.conv3(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.bn3(x)
        x = self.conv4(x, edge_index, edge_weight)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv5(x, edge_index, edge_weight)
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.25, training=self.training)
        gnn_score = self.fc2(x).view(-1)
        combined = torch.cat((torch.unsqueeze(gnn_score, 1), torch.unsqueeze(physics_score, 1)), dim=1)
        output = torch.squeeze(self.hybrid_score(combined))
        return output



def train(model, loader, optimizer, device):
    model.train()
    loss_all = 0
    total = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.edge_attr.view(-1), data.batch, data.physics_score)
        loss = F.mse_loss(output, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        total += data.num_graphs
        optimizer.step()
    return np.sqrt(loss_all / total)


@torch.no_grad()
def test(model, loader, device):
    model.eval()

    loss_all = 0
    total = 0

    y_true = []
    y_pred = []

    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.edge_attr.view(-1), data.batch, data.physics_score)
        loss = F.mse_loss(output, data.y)
        loss_all += loss.item() * data.num_graphs
        total += data.num_graphs
        y_true.extend(data.y.tolist())
        y_pred.extend(output.tolist())


    r_p = np.corrcoef(y_true, y_pred)[0,1]
    r_s = spearmanr(y_true, y_pred)[0]
    return np.sqrt(loss_all / total), r_p, r_s, y_true, y_pred

def plot_corr(y_true, y_pred, plot_dir):
    plt.clf()
    sns.scatterplot(y_true, y_pred)
    plt.xlabel('Actual -log(K)')
    plt.ylabel('Predicted -log(K)')
    plt.savefig(plot_dir)

def save_weights(model, weight_dir):
    torch.save(model.state_dict(), weight_dir)

def save_results(log_dir, split, mode, y_pred, loader):
    outfile = open(os.path.join(log_dir, f'{mode}_y_pred_{split}.pkl'), 'wb')
    pickle.dump(y_pred, outfile)

    codes = []
    for data in loader:
        codes.extend(data.pdb)

    outfile = open(os.path.join(log_dir, f'{mode}_loader_codes_{split}.pkl'), 'wb')
    pickle.dump(codes, outfile)


def train_pdbbind(split, architecture, mode, device, log_dir, data_path, split_path, seed=None):
    logger = logging.getLogger('pdbbind_log')

    num_epochs = 100
    batch_size = 700
    hidden_dim = 64
    learning_rate = 1e-4
    train_split = os.path.join(split_path, f'train_{split}.txt')
    val_split = os.path.join(split_path, f'val_{split}.txt')
    test_split = os.path.join(split_path,f'test_{split}.txt')
    train_loader = pdbbind_dataloader(batch_size, data_dir=data_path, split_file=train_split)
    f = open(os.path.join(log_dir, 'output.txt'), "a")
    f.write('Train size:' + str(len(train_loader)) + '\n')
    f.close()
    val_loader = pdbbind_dataloader(batch_size, data_dir=data_path, split_file=val_split)
    f = open(os.path.join(log_dir, 'output.txt'), "a")
    f.write('Val size:' + str(len(val_loader)) + '\n')
    f.close()
    test_loader = pdbbind_dataloader(500, data_dir=data_path, split_file=test_split)
    f = open(os.path.join(log_dir, 'output.txt'), "a")
    f.write('Test size:' + str(len(test_loader)) + '\n')
    f.close()

    if not os.path.exists(os.path.join(log_dir, 'params.txt')):
        with open(os.path.join(log_dir, 'params.txt'), 'w') as f:
            f.write(f'Split method: {split}\n')
            f.write(f'Model: {architecture}\n')
            f.write(f'Epochs: {num_epochs}\n')
            f.write(f'Batch size: {batch_size}\n')
            f.write(f'Hidden dim: {hidden_dim}\n')
            f.write(f'Learning rate: {learning_rate}')

    for data in train_loader:
        num_features = data.num_features
        break

    if architecture == 'GCN':
        model = GCN(num_features, hidden_dim=hidden_dim).to(device)
    elif architecture == 'GCN_score_layer':
        model = GCN_score_layer(num_features, hidden_dim=hidden_dim).to(device)
    model.to(device)

    if mode == "train_results":
        model.load_state_dict(torch.load(os.path.join(log_dir, f'best_weights_{split}.pt')))
        rmse, pearson, spearman, y_true, y_pred = test(model, train_loader, device)
        save_results(log_dir, split, "train", y_pred, train_loader)
    elif mode == "train":
        best_val_loss = 999


        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(1, num_epochs+1):
            start = time.time()
            train_loss = train(model, train_loader, optimizer, device)
            val_loss, r_p, r_s, y_true, y_pred = test(model, val_loader, device)
            if val_loss < best_val_loss:
                save_weights(model, os.path.join(log_dir, f'best_weights_{split}.pt'))
                plot_corr(y_true, y_pred, os.path.join(log_dir, f'corr_{split}.png'))
                best_val_loss = val_loss

            elapsed = (time.time() - start)
            f = open(os.path.join(log_dir, 'output.txt'), "a")
            f.write('Epoch: {:03d}, Time: {:.3f} s\n'.format(epoch, elapsed))
            f.close()
            print('Epoch: {:03d}, Time: {:.3f} s'.format(epoch, elapsed))
            f = open(os.path.join(log_dir, 'output.txt'), "a")
            f.write('\tTrain RMSE: {:.7f}, Val RMSE: {:.7f}, Pearson R: {:.7f}, Spearman R: {:.7f}\n'.format(train_loss, val_loss, r_p, r_s))
            f.close()
            print('\tTrain RMSE: {:.7f}, Val RMSE: {:.7f}, Pearson R: {:.7f}, Spearman R: {:.7f}'.format(train_loss, val_loss, r_p, r_s))
            logger.info('{:03d}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\n'.format(epoch, train_loss, val_loss, r_p, r_s))

            if epoch == num_epochs:
                save_results(log_dir, split, 'train', y_pred, train_loader)

    if mode == "train" or mode == "test":
        # save testing results
        test_file = os.path.join(log_dir, f'test_results_{split}.txt')
        model.load_state_dict(torch.load(os.path.join(log_dir, f'best_weights_{split}.pt')))
        rmse, pearson, spearman, y_true, y_pred = test(model, test_loader, device)
        plot_corr(y_true, y_pred, os.path.join(log_dir, f'corr_{split}_test.png'))
        f = open(os.path.join(log_dir, 'output.txt'), "a")
        f.write('Test RMSE: {:.7f}, Pearson R: {:.7f}, Spearman R: {:.7f}\n'.format(rmse, pearson, spearman))
        f.close()
        print('Test RMSE: {:.7f}, Pearson R: {:.7f}, Spearman R: {:.7f}'.format(rmse, pearson, spearman))
        with open(test_file, 'a+') as out:
            out.write('{}\t{:.7f}\t{:.7f}\t{:.7f}\n'.format(seed, rmse, pearson, spearman))

        save_results(log_dir, split, 'test', y_pred, test_loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir', type=str, help='directory where all logging data will be written')
    parser.add_argument('root', type=str, help='directory where raw and processed directories can be found')
    parser.add_argument('split_path', type=str, help='directory where raw and processed directories can be found')
    parser.add_argument('--mode', type=str, default='train', help='either train, test, or train_results')
    parser.add_argument('--split', type=str, default='random', help='name of split files')
    parser.add_argument('--architecture', type=str, default='GCN', help='either GCN or GCN_score_layer')
    parser.add_argument('--log_dir', type=str, default=None, help='specific logging directory')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_dir = args.log_dir


    if args.mode == 'train':
        if log_dir is None:
            now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            log_dir = os.path.join(args.out_dir, 'logs', now)
        else:
            log_dir = os.path.join(args.out_dir, 'logs', log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        train_pdbbind(args.split, args.architecture, args.mode, device, log_dir, args.root, args.split_path)
    elif args.mode == 'test' or args.mode == 'train_results':
        seed = 0
        log_dir = os.path.join(args.out_dir, 'logs', args.log_dir)
        np.random.seed(seed)
        torch.manual_seed(seed)
        train_pdbbind(args.split, args.architecture, args.mode, device, log_dir, args.root, args.split_path, seed)

if __name__=="__main__":
    main()






