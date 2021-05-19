import time
import argparse
from tqdm import tqdm

import torch
import numpy as np
import pandas as pd

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
    parser.add_argument('--p_batch_size', type=int, default=40000)
    parser.add_argument('--mini_graph', type=bool, default=False)
    args = parser.parse_args()
    print(args)

    torch.manual_seed(12345)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    dataset = MAG240MDataset(ROOT)
    evaluator = MAG240MEvaluator()

    train_idx = dataset.get_idx_split('train')
    valid_idx = dataset.get_idx_split('valid')
    test_idx = dataset.get_idx_split('test')

    ap_edge = np.load(f'{dataset.dir}/sorted_author_paper_edge.npy')
    idx = np.concatenate([train_idx,valid_idx],0)

    year = dataset.paper_year
    label = dataset.paper_label

    paper_author_list = []
    author_list = []
    bias = 0
    print('--reading paper author list--')
    for i in tqdm(range(idx.shape[0])):
        i = idx[i]
        for j in range(bias, ap_edge.shape[1]):
            if i == ap_edge[1,j]:
                paper_author_list.append([i,ap_edge[0,j]])
                author_list.append(ap_edge[0,j])
            if i < ap_edge[1,j]:
                bias = j
                break
    author_list = np.unique(author_list)

    ap_edge = dataset.edge_index('author', 'writes', 'paper')

    print('--reading author paper list--')
    bias = 0
    author_paper_list = []
    paper_list = []
    for i in tqdm(range(author_list.shape[0])):
        i = author_list[i]
        for j in tqdm(range(bias, ap_edge.shape[1])):
            if i == ap_edge[0, j]:
                author_paper_list.append([i,ap_edge[j, 1]])
                paper_list.append(ap_edge[1, j])
            if i < ap_edge[0, j]:
                bias = j
                break
    paper_list = np.unique(paper_list)


    related_year = year[paper_list]
    related_label = label[paper_list]

    target1 = pd.DataFrame(author_paper_list,columns=['author','ind'])
    target2 = pd.DataFrame(np.concatenate([paper_list.reshape(-1,1),np.array(related_year).reshape(-1,1),np.array(related_label).reshape(-1,1)],1),columns=['ind','year','label'])
    target2 = target2[target2.year<2019]

    target = pd.merge(target1,target2)

    result = []
    print('resulting...')
    ind = list(target.groupby('author').year.max().index)
    val = target.groupby('author').year.max().values
    r = np.concatenate([np.array(ind).reshape(-1, 1), val.reshape(-1, 1)], 1)

    a_ = []
    for i in range(r.shape[0]):
        tmp = target[(target.author == r[i, 0]) & (target.year == r[i, 1])]
        try:
            a_.append([tmp.author.values[0], tmp.label.mode().values[0]])
        except:
            a_.append([tmp.author.values[0], -1])

    target1 = pd.DataFrame(paper_author_list,columns=['ind','author'])
    target2 = pd.DataFrame(a_,columns=['author','label'])

    target = pd.merge(target1,target2)
    target = pd.DataFrame(target.groupby('ind').label.agg(lambda x: x.value_counts().index[0]))
    y_pred = target.loc[valid_idx].label.values
    y_true = label[valid_idx]

    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.from_numpy(y_pred)
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.from_numpy(y_true)

    assert (y_true.numel() == y_pred.numel())
    assert (y_true.dim() == y_pred.dim() == 1)

    acc = int((y_true == y_pred).sum()) / y_true.numel()
    print(acc)






