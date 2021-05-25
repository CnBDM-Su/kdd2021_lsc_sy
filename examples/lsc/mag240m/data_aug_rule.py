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

ap_edge = np.load(f'{dataset.dir}/sorted_author_paper_edge.npy')
a_l = {}
bias = 0
for i in tqdm(range(train_idx.shape[0])):
    i = train_idx[i].numpy()
    for j in range(bias,ap_edge.shape[1]):
        print(i,ap_edge[0,j])
        if i==ap_edge[0,j]:
            if ap_edge[0,j] not in a_l.keys():
                a_l[ap_edge[0,j]] = [paper_label[ap_edge[1,j]]]
            else:
                a_l[ap_edge[0, j]].append(paper_label[ap_edge[1,j]])
        elif i<ap_edge[0,j]:
            bias = j
            break
print(len(a_l.keys()))
for i in a_l.keys():
    if len(a_l[i]) > 1:
        print(a_l[i])
