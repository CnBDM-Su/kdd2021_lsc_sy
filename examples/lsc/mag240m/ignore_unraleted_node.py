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
    args = parser.parse_args()
    print(args)

    torch.manual_seed(12345)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    dataset = MAG240MDataset(ROOT)
    evaluator = MAG240MEvaluator()

    train_idx = dataset.get_idx_split('train')
    valid_idx = dataset.get_idx_split('valid')
    test_idx = dataset.get_idx_split('test')

    meaningful_idx = np.concatenate([train_idx, valid_idx, test_idx], 0)
    meaningful_idx = np.sort(meaningful_idx)

    path = f'{dataset.dir}/paper_to_paper_symmetric.pt'
    if osp.exists(path):
        adj_t = torch.load(path)

    for i in meaningful_idx:


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
