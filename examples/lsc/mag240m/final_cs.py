# NOTE: 256GB CPU memory required to run this script.

import os.path as osp
import time
from copy import deepcopy
import torch
import numpy as np
from torch_sparse import SparseTensor
from torch_geometric.nn import CorrectAndSmooth
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from ogb.lsc import MAG240MDataset, MAG240MEvaluator
from ogb.utils.url import makedirs

def kdd_cs(dataset, train_idx, valid_idx, test_idx, paper_label,
       num_correction_layers, correction_alpha, num_smoothing_layers, smoothing_alpha):

    evaluator = MAG240MEvaluator()

    save_path = 'results/final_rgat_result'

    print('Reading MLP soft prediction...', end=' ', flush=True)
    t = time.perf_counter()
    y_pred = torch.from_numpy(np.load(save_path+'/r.npz')['y_pred'])
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
    # print('graph_size',adj_t)

    y_train = torch.from_numpy(paper_label[train_idx]).to(torch.long)
    y_valid = torch.from_numpy(paper_label[valid_idx]).to(torch.long)

    def train(smoothing_alpha, y_pred=y_pred, num_correction_layers=num_correction_layers,
              correction_alpha=correction_alpha, num_smoothing_layers=num_smoothing_layers):
        model = CorrectAndSmooth(num_correction_layers, correction_alpha,
                                 num_smoothing_layers, smoothing_alpha,
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
        return y_pred, valid_acc

    y_pred_tmp, valid_acc = train(smoothing_alpha)
    y_pred_best = y_pred_tmp
    y_pred = y_pred_best
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

    res = {'y_pred': y_pred[test_idx].argmax(dim=-1)}
    evaluator.save_test_submission(res, save_path)
