import time
import argparse
from tqdm import tqdm

import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn import ModuleList, Linear, BatchNorm1d, Identity
import random
import os.path as osp

from ogb.utils.url import makedirs

from ogb.lsc import MAG240MDataset, MAG240MEvaluator
import sys
from torch_sparse import SparseTensor
sys.path.append('/var/ogb/ogb/lsc')
from mag240m_mini_graph import MAG240MMINIDataset
from root import ROOT


class MLP(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int, dropout: float = 0.0,
                 batch_norm: bool = True, relu_last: bool = False):
        super(MLP, self).__init__()

        self.lins = ModuleList()
        self.lins.append(Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.lins.append(Linear(hidden_channels, out_channels))

        self.batch_norms = ModuleList()
        for _ in range(num_layers - 1):
            norm = BatchNorm1d(hidden_channels) if batch_norm else Identity()
            self.batch_norms.append(norm)

        self.dropout = dropout
        self.relu_last = relu_last

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for batch_norm in self.batch_norms:
            batch_norm.reset_parameters()

    def forward(self, x):
        for lin, batch_norm in zip(self.lins[:-1], self.batch_norms):
            x = lin(x)
            if self.relu_last:
                x = batch_norm(x).relu_()
            else:
                x = batch_norm(x.relu_())
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


def train(model, x_train, y_train, batch_size, optimizer):
    model.train()

    total_loss = 0
    for idx in DataLoader(range(y_train.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x_train[idx]), y_train[idx])
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * idx.numel()

    return total_loss / y_train.size(0)


@torch.no_grad()
def test(model, x_eval, y_eval, evaluator):
    model.eval()
    y_pred = model(x_eval).argmax(dim=-1)
    return evaluator.eval({'y_true': y_eval, 'y_pred': y_pred})['acc']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--hidden_channels', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=2),
    parser.add_argument('--no_batch_norm', action='store_true')
    parser.add_argument('--relu_last', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=380000)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--evaluate', type=bool, default=False)
    parser.add_argument('--p_batch_size', type=int, default=40000)
    parser.add_argument('--mini_graph', type=bool, default=False)
    parser.add_argument('--parallel', type=bool, default=False)
    args = parser.parse_args()
    print(args)

    torch.manual_seed(12345)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    evaluator = MAG240MEvaluator()

    if args.mini_graph:
        dataset = MAG240MMINIDataset(ROOT)
    else:
        dataset = MAG240MDataset(ROOT)

    train_idx = dataset.get_idx_split('train')
    valid_idx = dataset.get_idx_split('valid')
    test_idx = dataset.get_idx_split('test')

    paper_label = dataset.paper_label

    path = f'{dataset.dir}/sorted_weighted_author_paper_edge.npy'
    if not osp.exists(path):
        print('Generating sorted weighted author paper edges...')
        t = time.perf_counter()
        ap_edge = np.load(f'{dataset.dir}/weighted_author_paper_edge.npy')
        ap_edge = ap_edge[:, ap_edge[1, :].argsort()]
        np.save(path, ap_edge)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

    path = f'{dataset.dir}/paper_relation_weighted_feat.npy'
    if not osp.exists(path):
        print('Generating paper relation weighted features...')
        t = time.perf_counter()
        #N = dataset.num_papers + dataset.num_authors + dataset.num_institutions
        x = np.load(f'{dataset.dir}/full_weighted_feat.npy')
        weighted_edge = np.load(f'{dataset.dir}/sorted_weighted_author_paper_edge.npy')
        # y = np.zeros(shape=(dataset.num_papers, 1536))
        # y = np.memmap(path, dtype=np.float16, mode='w+',
        #               shape=(dataset.num_papers, 1536))
        row, col, val = torch.from_numpy(weighted_edge)
        adj_t = SparseTensor(
            row=col.long(), col=row.long(), value=val.float(),
            sparse_sizes=(dataset.num_papers, dataset.num_authors),
            is_sorted=True)

        inputs = torch.from_numpy(x[dataset.num_papers:dataset.num_papers+dataset.num_authors]).float()
        outputs = adj_t.matmul(inputs, reduce='mean').numpy()
        x = np.concatenate([x[:dataset.num_papers],outputs],1)
        np.save(path, x)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')
    else:
        x = np.load(path)

    if args.evaluate == False:
        t = time.perf_counter()

        print('Reading training node features...', end=' ', flush=True)
        # x_train = dataset.paper_feat[train_idx]
        x_train = x[train_idx]
        x_train = torch.from_numpy(x_train).to(torch.float).to(device)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')
        t = time.perf_counter()
        print('Reading validation node features...', end=' ', flush=True)
        x_valid = x[valid_idx]
        # x_valid = dataset.paper_feat[valid_idx]
        x_valid = torch.from_numpy(x_valid).to(torch.float).to(device)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

        y_train = torch.from_numpy(paper_label[train_idx])
        y_train = y_train.to(device, torch.long)
        y_valid = torch.from_numpy(paper_label[valid_idx])
        y_valid = y_valid.to(device, torch.long)

        if args.mini_graph:
            save_path = 'results/mini_cs_weighted'
        else:
            save_path = 'results/cs'
        makedirs(save_path)
        model = MLP(dataset.num_paper_features * 2, args.hidden_channels,
                    dataset.num_classes, args.num_layers, args.dropout,
                    not args.no_batch_norm, args.relu_last).to(device)
        if args.parallel == True:
            model = torch.nn.DataParallel(model, device_ids=[0,1,2,3,4,5,6,7])
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        num_params = sum([p.numel() for p in model.parameters()])
        print(f'#Params: {num_params}')

        best_valid_acc = 0
        for epoch in range(1, args.epochs + 1):
            loss = train(model, x_train, y_train, args.batch_size, optimizer)
            train_acc = test(model, x_train, y_train, evaluator)
            valid_acc = test(model, x_valid, y_valid, evaluator)
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                torch.save(model.state_dict(), save_path + '/model.pt')
            if epoch % 100 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
                      f'Train: {train_acc:.4f}, Valid: {valid_acc:.4f}, '
                      f'Best: {best_valid_acc:.4f}')
    else:
        model = MLP(dataset.num_paper_features * 2, args.hidden_channels,
                    dataset.num_classes, args.num_layers, args.dropout,
                    not args.no_batch_norm, args.relu_last).to(device)
    model.load_state_dict(torch.load(save_path + '/model.pt'))
    model.eval()

    pbar = tqdm(total=dataset.num_papers)
    pbar.set_description('Saving model predictions')

    out = []
    for i in range(0, dataset.num_papers, args.batch_size):
        x_ = x[i:min(i + args.batch_size, dataset.num_papers)]
        x_ = torch.from_numpy(x_).to(torch.float).to(device)
        with torch.no_grad():
            out.append(model(x_).softmax(dim=-1).cpu().numpy())
        pbar.update(x_.size(0))
    pbar.close()
    np.save(save_path + '/pred.npy', np.concatenate(out, axis=0))
    # np.savez('results/cs/pred.npz', *out)
