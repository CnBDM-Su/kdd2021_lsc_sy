# NOTE: 256GB CPU memory required to run this script.

import os.path as osp
import time
import argparse

import torch
import numpy as np
from torch_sparse import SparseTensor
# from torch_geometric.nn import CorrectAndSmooth
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from ogb.lsc import MAG240MDataset, MAG240MEvaluator
from root import ROOT
from torch_geometric.typing import Adj, OptTensor

from torch import Tensor
import torch.nn.functional as F

from torch_geometric.nn.models import LabelPropagation


class CorrectAndSmooth(torch.nn.Module):

    def __init__(self, num_correction_layers: int, correction_alpha: float,
                 num_smoothing_layers: int, smoothing_alpha: float,
                 autoscale: bool = True, scale: float = 1.0):
        super(CorrectAndSmooth, self).__init__()
        self.autoscale = autoscale
        self.scale = scale

        self.prop1 = LabelPropagation(num_correction_layers, correction_alpha)
        self.prop2 = LabelPropagation(num_smoothing_layers, smoothing_alpha)

    def correct(self, y_soft: Tensor, y_true: Tensor, mask: Tensor,
                edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:

        assert abs((float(y_soft.sum()) / y_soft.size(0)) - 1.0) < 1e-2

        numel = int(mask.sum()) if mask.dtype == torch.bool else mask.size(0)
        assert y_true.size(0) == numel

        if y_true.dtype == torch.long:
            y_true = F.one_hot(y_true.view(-1), y_soft.size(-1))
            y_true = y_true.to(y_soft.dtype)

        error = torch.zeros_like(y_soft)
        error[mask] = y_true - y_soft[mask]

        if self.autoscale:
            smoothed_error = self.prop1(error, edge_index,
                                        edge_weight=edge_weight,
                                        post_step=lambda x: x.clamp_(-1., 1.))

            sigma = error[mask].abs().sum() / numel
            scale = sigma / smoothed_error.abs().sum(dim=1, keepdim=True)
            scale[scale.isinf() | (scale > 1000)] = 1.0
            return y_soft + scale * smoothed_error
        else:

            def fix_input(x):
                x[mask] = error[mask]
                return x

            smoothed_error = self.prop1(error, edge_index,
                                        edge_weight=edge_weight,
                                        post_step=fix_input)
            return y_soft + self.scale * smoothed_error

    def smooth(self, y_soft: Tensor, y_true: Tensor, mask: Tensor,
               edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:

        numel = int(mask.sum()) if mask.dtype == torch.bool else mask.size(0)
        assert y_true.size(0) == numel

        if y_true.dtype == torch.long:
            y_true = F.one_hot(y_true.view(-1), y_soft.size(-1))
            y_true = y_true.to(y_soft.dtype)

        y_soft[mask] = y_true

        return self.prop2(y_soft, edge_index, edge_weight=edge_weight)

    def __repr__(self):
        L1, alpha1 = self.prop1.num_layers, self.prop1.alpha
        L2, alpha2 = self.prop2.num_layers, self.prop2.alpha
        return (f'{self.__class__.__name__}(\n'
                f'    correct: num_layers={L1}, alpha={alpha1}\n'
                f'    smooth:  num_layers={L2}, alpha={alpha2}\n'
                f'    autoscale={self.autoscale}, scale={self.scale}\n'
                ')')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_correction_layers', type=int, default=3)
    parser.add_argument('--correction_alpha', type=float, default=1.0)
    parser.add_argument('--num_smoothing_layers', type=int, default=2)
    parser.add_argument('--smoothing_alpha', type=float, default=0.8)
    args = parser.parse_args()
    print(args)

    dataset = MAG240MDataset(ROOT)
    evaluator = MAG240MEvaluator()

    print('Reading MLP soft prediction...', end=' ', flush=True)
    t = time.perf_counter()
    # y_pred = torch.from_numpy(np.load('results/cs/pred.npy'))
    y_pred = torch.rand(121000000, 768)
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

    train_idx = torch.from_numpy(dataset.get_idx_split('train'))
    valid_idx = torch.from_numpy(dataset.get_idx_split('valid'))
    test_idx = torch.from_numpy(dataset.get_idx_split('test'))

    y_train = torch.from_numpy(dataset.paper_label[train_idx]).to(torch.long)
    y_valid = torch.from_numpy(dataset.paper_label[valid_idx]).to(torch.long)

    # model = CorrectAndSmooth(args.num_correction_layers, args.correction_alpha,
    #                          args.num_smoothing_layers, args.smoothing_alpha,
    #                          autoscale=True)

    t = time.perf_counter()
    print('Correcting predictions...', end=' ', flush=True)
    assert abs((float(y_pred.sum()) / y_pred.size(0)) - 1.0) < 1e-2

    numel = int(train_idx.sum()) if train_idx.dtype == torch.bool else train_idx.size(0)
    assert y_train.size(0) == numel

    if y_train.dtype == torch.long:
        y_true = F.one_hot(y_train.view(-1), y_pred.size(-1))
        y_true = y_true.to(y_pred.dtype)

    y_shape = y_pred.shape
    error_ = y_train - y_pred[train_idx]
    del y_pred
    error = torch.zeros(y_shape)
    error[train_idx] = error_
    # error = torch.zeros_like(y_pred)
    # error[train_idx] = y_train - y_pred[train_idx]
    prop1 = LabelPropagation(args.num_correction_layers, args.correction_alpha)
    edge_weight = None
    smoothed_error = prop1(error, adj_t,
                                edge_weight=edge_weight,
                                post_step=lambda x: x.clamp_(-1., 1.))

    sigma = error[train_idx].abs().sum() / numel
    scale = sigma / smoothed_error.abs().sum(dim=1, keepdim=True)
    scale[scale.isinf() | (scale > 1000)] = 1.0

    print('Reading MLP soft prediction...', end=' ', flush=True)
    t = time.perf_counter()
    y_pred = torch.from_numpy(np.load('results/cs/pred.npy'))
    print(f'Done! [{time.perf_counter() - t:.2f}s]')

    y_pred = y_pred + scale * smoothed_error
    # y_pred = model.correct(y_pred, y_train, train_idx, adj_t)
    print(f'Done! [{time.perf_counter() - t:.2f}s]')

    t = time.perf_counter()
    print('Smoothing predictions...', end=' ', flush=True)
    numel = int(train_idx.sum()) if train_idx.dtype == torch.bool else train_idx.size(0)
    assert y_train.size(0) == numel

    if y_train.dtype == torch.long:
        y_train = F.one_hot(y_train.view(-1), y_pred.size(-1))
        y_train = y_train.to(y_pred.dtype)

    y_pred[train_idx] = y_train

    prop2 = LabelPropagation(args.num_smoothing_layers, args.smoothing_alpha)

    y_pred = prop2(y_pred, adj_t, edge_weight=edge_weight)
    # y_pred = model.smooth(y_pred, y_train, train_idx, adj_t)
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

    res = {'y_pred': y_pred[test_idx].argmax(dim=-1)}
    evaluator.save_test_submission(res, 'results/cs')
