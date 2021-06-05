# NOTE: 256GB CPU memory required to run this script.

import os.path as osp
import time
import argparse
from copy import deepcopy
import torch
import numpy as np
from torch_sparse import SparseTensor
from torch_geometric.nn import CorrectAndSmooth
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from ogb.lsc import MAG240MDataset, MAG240MEvaluator
from root import ROOT
from ogb.utils.url import makedirs
import sys
sys.path.append('/var/ogb/ogb/lsc')
from mag240m_mini_graph import MAG240MMINIDataset
class MAG240MEvaluator:
    def eval(self, input_dict):
        assert 'y_pred' in input_dict and 'y_true' in input_dict

        y_pred, y_true = input_dict['y_pred'], input_dict['y_true']

        if not isinstance(y_pred, torch.Tensor):
            y_pred = torch.from_numpy(y_pred)
        if not isinstance(y_true, torch.Tensor):
            y_true = torch.from_numpy(y_true)

        assert (y_true.numel() == y_pred.numel())
        assert (y_true.dim() == y_pred.dim() == 1)

        return {'acc': int((y_true == y_pred).sum()) / y_true.numel()}

    def save_test_submission(self, input_dict, dir_path):
        # assert 'y_pred' in input_dict
        y_pred = input_dict['y_pred']
        assert y_pred.shape == (146818, )

        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        y_pred = y_pred.astype(np.short)

        makedirs(dir_path)
        filename = osp.join(dir_path, 'y_pred_mag240m')
        np.savez_compressed(filename, y_pred=y_pred)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_correction_layers', type=int, default=3)
    parser.add_argument('--correction_alpha', type=float, default=1.0)
    parser.add_argument('--num_smoothing_layers', type=int, default=2)
    parser.add_argument('--smoothing_alpha', type=float, default=0.8)
    parser.add_argument('--mini_graph', type=bool, default=False)
    args = parser.parse_args()
    print(args)

    evaluator = MAG240MEvaluator()

    if args.mini_graph:
        dataset = MAG240MMINIDataset(ROOT)
        # save_path = 'results/mini_cs_weighted'
        save_path = 'results/rgat_cs_new'

    else:
        dataset = MAG240MDataset(ROOT)
        save_path = 'results/cs'

    train_idx = torch.from_numpy(dataset.get_idx_split('train'))
    valid_idx = torch.from_numpy(dataset.get_idx_split('valid'))
    test_idx = torch.from_numpy(dataset.get_idx_split('test'))
    paper_label = dataset.paper_label

    print('Reading MLP soft prediction...', end=' ', flush=True)
    t = time.perf_counter()
    y_pred = torch.from_numpy(np.load('results/rgat_cs_new/256ap_rgat.npz')['y_pred'])
    # y_pred = torch.from_numpy(np.load(save_path+'/rgat_pred.npz')['y_pred'])
    # y_pred = torch.from_numpy(np.load(save_path+'/pred.npy'))
    print(f'Done! [{time.perf_counter() - t:.2f}s]')

    t = time.perf_counter()
    print('Reading adjacency matrix...', end=' ', flush=True)
    path = f'{dataset.dir}/paper_to_paper_symmetric_gcn.pt'
    if osp.exists(path):
        adj_t = torch.load(path)
    else:
        path_sym = f'{dataset.dir}/paper_to_paper_symmetric.pt'
        if osp.exists(path_sym):
            adj_t = torch.load(path_sym)
        else:
            edge_index = dataset.edge_index('paper', 'cites', 'paper')
            edge_index = torch.from_numpy(edge_index)
            adj_t = SparseTensor(
                row=edge_index[0], col=edge_index[1],
                sparse_sizes=(dataset.num_papers, dataset.num_papers),
                is_sorted=True)
            adj_t = adj_t.to_symmetric()
            torch.save(adj_t, path_sym)
        adj_t = gcn_norm(adj_t, add_self_loops=True)
        torch.save(adj_t, path)
    print(f'Done! [{time.perf_counter() - t:.2f}s]')

    y_train = torch.from_numpy(paper_label[train_idx]).to(torch.long)
    y_valid = torch.from_numpy(paper_label[valid_idx]).to(torch.long)
    y_test = torch.from_numpy(paper_label[test_idx]).to(torch.long)
    # edge_index = np.load(f'{dataset.dir}/weighted_paper_paper_edge.npy')
    # edge_index = torch.from_numpy(edge_index)
    # adj_t = adj_t.set_value(edge_index[2], layout='coo')

    def train(smoothing_alpha, y_pred=y_pred):

        model = CorrectAndSmooth(args.num_correction_layers, args.correction_alpha,
                                 args.num_smoothing_layers, smoothing_alpha,
                                 autoscale=True)

        t = time.perf_counter()
        y_pred = deepcopy(y_pred)

        print('Correcting predictions...', end=' ', flush=True)
        assert abs((float(y_pred.sum()) / y_pred.size(0)) - 1.0) < 1e-2

        numel = int(train_idx.sum()) if train_idx.dtype == torch.bool else train_idx.size(0)
        assert y_train.size(0) == numel

        y_pred = model.correct(y_pred, y_train, train_idx, adj_t)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')
        t = time.perf_counter()

        print('Smoothing predictions...', end=' ', flush=True)
        y_pred = model.smooth(y_pred, y_train, train_idx, adj_t)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

        train_acc = evaluator.eval({
            'y_true': y_train,
            'y_pred': y_pred[train_idx].argmax(dim=-1)
        })['acc']
        valid_acc = evaluator.eval({
            'y_true': y_valid,
            'y_pred': y_pred[valid_idx].argmax(dim=-1)
        })['acc']
        print(f'Train: {train_acc:.4f}, Valid: {valid_acc:.4f}')
        return y_pred, train_acc
    acc_lis = []
    alpha_lis = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
    for i in alpha_lis:
        y_pred_tmp, train_acc = train(i)
        acc_lis.append(train_acc)
        if len(acc_lis)>1:
            if acc_lis[-1]-acc_lis[-2] > 0.0001:
                y_pred_best = y_pred_tmp
            else:
                try:
                    y_pred = y_pred_best
                except:
                    y_pred = y_pred_tmp
                break
    print('smooth alpha is',i)
    train_acc = evaluator.eval({
        'y_true': y_train,
        'y_pred': y_pred[train_idx].argmax(dim=-1)
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_valid,
        'y_pred': y_pred[valid_idx].argmax(dim=-1)
    })['acc']
    print(f'Train: {train_acc:.4f}, Valid: {valid_acc:.4f}')

    # np.save('results/rgat_cs/rgat_cs_pred.npy',y_pred)

    res = {'y_pred': y_pred[test_idx].argmax(dim=-1)}
    evaluator.save_test_submission(res, save_path)
