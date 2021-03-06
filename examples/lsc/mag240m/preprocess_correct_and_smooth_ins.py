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
    parser.add_argument('--a_batch_size', type=int, default=40000)
    args = parser.parse_args()
    print(args)

    torch.manual_seed(12345)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    dataset = MAG240MDataset(ROOT)
    evaluator = MAG240MEvaluator()

    train_idx = dataset.get_idx_split('train')
    valid_idx = dataset.get_idx_split('valid')
    test_idx = dataset.get_idx_split('test')

    path = f'{dataset.dir}/sorted_author_paper_edge.npy'
    if not osp.exists(path):
        print('Generating sorted author paper edges...')
        t = time.perf_counter()
        ap_edge = dataset.edge_index('author', 'writes', 'paper')
        ap_edge = ap_edge[:, ap_edge[1, :].argsort()]
        np.save(path, ap_edge)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

    path = f'{dataset.dir}/author_relation_feat.npy'
    if not osp.exists(path):
        print('Generating author relation features...')
        t = time.perf_counter()
        N = dataset.num_papers + dataset.num_authors + dataset.num_institutions
        x = np.memmap(f'{dataset.dir}/full_feat.npy', dtype=np.float16,
                      mode='r', shape=(N, 768))
        y = np.memmap(path, dtype=np.float16, mode='w+',
                      shape=(dataset.num_papers, 771))
        ai_edge = dataset.edge_index('author', 'affiliated_with', 'institution')
        bias = 0
        a_batch_size = args.a_batch_size
        zero_i = 0
        one_i = 0
        two_i = 0
        three_i = 0
        for a_batch in tqdm(range(dataset.num_authors // a_batch_size)):
            fea_ = []
            end = min((a_batch + 1) * a_batch_size, dataset.num_authors)
            for i in range(a_batch * a_batch_size, end):
                sign = 0
                fea = []
                for j in range(bias, len(ai_edge[0])):
                    if ai_edge[0, j] == i:
                        fea.append(ai_edge[1, j])
                        sign = 1
                    if len(fea) > 2:
                        break
                    if (sign == 1) and (ai_edge[0, j] != i):
                        break
                bias = j
                if len(fea)==0:
                    fea = np.array([-1, -1, -1])
                    zero_i +=1
                elif len(fea)==1:
                    fea = np.array([fea[0],-1,-1])
                    one_i +=1
                elif len(fea)==2:
                    fea = np.concatenate([np.sort(fea),np.array([-1])],0)
                    two_i += 1
                else:
                    fea = np.sort(fea)
                    three_i += 1
                # fea = x[fea]
                fea_.append(fea)
            fea_ = np.array(fea_)
            print(fea_.shape)

            y[a_batch * a_batch_size:end] = np.concatenate(
                [x[(a_batch * a_batch_size)+dataset.num_papers:end + dataset.num_papers], fea_], 1)
        print('zero institute auther num:', one_i)
        print('one institute auther num:',one_i)
        print('two institute auther num:', two_i)
        print('three institute auther num:', three_i)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

    path = f'{dataset.dir}/paper_relation_feat_2.npy'
    if not osp.exists(path):
        print('Generating paper relation features...')
        t = time.perf_counter()
        x = np.memmap(f'{dataset.dir}/author_relation_feat.npy', dtype=np.float16,
                           mode='r', shape=(dataset.num_authors, 1536))
        y = np.memmap(path, dtype=np.float16, mode='w+',
                      shape=(dataset.num_papers, 2304))
        ap_edge = np.load(f'{dataset.dir}/sorted_author_paper_edge.npy')
        bias = 0
        p_batch_size = args.p_batch_size
        for p_batch in tqdm(range(dataset.num_papers//p_batch_size)):
            fea_ = []
            end = min((p_batch + 1) * p_batch_size, dataset.num_papers)
            for i in range(p_batch*p_batch_size, end):
                sign = 0
                fea = []
                for j in range(bias,len(ap_edge[0])):
                    if ap_edge[1,j] == i:
                        fea.append(ap_edge[0,j])
                        sign = 1
                    if len(fea) > 20:
                        break
                    if (sign==1) and (ap_edge[1,j] != i):
                        break
                bias = j
                fea = x[fea]
                fea_.append(np.mean(fea, 0))
            fea_ = np.array(fea_)
            y[p_batch*p_batch_size:end] = np.concatenate([x[p_batch*p_batch_size:end],fea_],1)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')


    x_fr = np.memmap(path, dtype=np.float16, mode='r',
                  shape=(dataset.num_papers, 2304))

    if args.evaluate==False:
        t = time.perf_counter()

        print('Reading training node features...', end=' ', flush=True)
        # x_train = dataset.paper_feat[train_idx]
        x_train = x_fr[train_idx]
        x_train = torch.from_numpy(x_train).to(torch.float).to(device)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')
        t = time.perf_counter()
        print('Reading validation node features...', end=' ', flush=True)
        x_valid = x_fr[valid_idx]
        # x_valid = dataset.paper_feat[valid_idx]
        x_valid = torch.from_numpy(x_valid).to(torch.float).to(device)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

        y_train = torch.from_numpy(dataset.paper_label[train_idx])
        y_train = y_train.to(device, torch.long)
        y_valid = torch.from_numpy(dataset.paper_label[valid_idx])
        y_valid = y_valid.to(device, torch.long)

        makedirs('results/cs')
        model = MLP(dataset.num_paper_features, args.hidden_channels,
                    dataset.num_classes, args.num_layers, args.dropout,
                    not args.no_batch_norm, args.relu_last).to(device)

        if torch.cuda.is_available():
            gpus = [4,5,6,7]
            model = model = torch.nn.DataParallel(model, device_ids=gpus)

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
                torch.save(model.state_dict(), 'results/cs/model.pt')
            if epoch % 100 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
                      f'Train: {train_acc:.4f}, Valid: {valid_acc:.4f}, '
                      f'Best: {best_valid_acc:.4f}')
    else:
        model = MLP(dataset.num_paper_features, args.hidden_channels,
                    dataset.num_classes, args.num_layers, args.dropout,
                    not args.no_batch_norm, args.relu_last).to(device)
    model.load_state_dict(torch.load('results/cs/model.pt'))
    model.eval()

    pbar = tqdm(total=dataset.num_papers)
    pbar.set_description('Saving model predictions')

    out = []
    for i in range(0, dataset.num_papers, args.batch_size):
        x = x_fr[i:min(i + args.batch_size, dataset.num_papers)]
        x = torch.from_numpy(x).to(torch.float).to(device)
        with torch.no_grad():
            out.append(model(x).softmax(dim=-1).cpu().numpy())
        pbar.update(x.size(0))
    pbar.close()
    np.save('results/cs/pred.npy', np.concatenate(out, axis=0))
    # np.savez('results/cs/pred.npz', *out)
