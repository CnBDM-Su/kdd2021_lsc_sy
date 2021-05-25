import os.path as osp
import time
import argparse
from tqdm import tqdm
import torch
import numpy as np
from torch_sparse import SparseTensor
from torch_geometric.nn import CorrectAndSmooth
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from ogb.lsc import MAG240MDataset
from root import ROOT
from ogb.utils.url import makedirs
import sys
sys.path.append('/var/ogb/ogb/lsc')
from mag240m_mini_graph import MAG240MMINIDataset


dataset = MAG240MMINIDataset(ROOT)

train_idx = torch.from_numpy(dataset.get_idx_split('train'))
valid_idx = torch.from_numpy(dataset.get_idx_split('valid'))
test_idx = torch.from_numpy(dataset.get_idx_split('test'))
paper_label = dataset.paper_label

ap_edge = dataset.edge_index('author', 'writes', 'paper')
a_l = {}
bias = 0
for i in tqdm(range(train_idx.shape[0])):
    print(train_idx[i].numpy())
    i = train_idx[i].numpy()
    for j in range(bias,ap_edge.shape[1]):
        print(i,ap_edge[0,i])
        if i==ap_edge[0,i]:
            if ap_edge[0,i] not in a_l.keys():
                a_l[ap_edge[0,i]] = [paper_label[ap_edge[1,i]]]
            else:
                a_l[ap_edge[0, i]].append(paper_label[ap_edge[1,i]])
        else:
            bias = j
            break
print(len(a_l.keys()))
for i in a_l.keys():
    if len(a_l[i]) > 1:
        print(a_l[i])
