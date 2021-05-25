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
for i in tqdm(range(ap_edge.shape[1])):
    if ap_edge[1,i] in train_idx:
        print(ap_edge[0,i],ap_edge[1,i])
        if ap_edge[0,i] not in a_l.keys():
            print(paper_label[ap_edge[1, i]])
            a_l[ap_edge[0,i]] = [paper_label[ap_edge[1,i]]]
        else:
            print(a_l[ap_edge[0, i]])
            a_l[ap_edge[0, i]] = a_l[ap_edge[0, i]].append(paper_label[ap_edge[1,i]])

for i in a_l.keys():
    if len(a_l[i]) > 3:
        print(a_l[i])
