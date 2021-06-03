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

dataset = MAG240MMINIDataset(ROOT)
co_author = np.load(f'{dataset.dir}/author_connect_graph.npy')
paper = np.load(f'{dataset.dir}/paper_connect_graph.npy')
sorted_paper = paper[:, paper[1, :].argsort()]
bias = 0
overlap = []
for i in range(co_author.shape[1]):
    for j in range(bias, paper.shape[1]):
        if co_author[0,i]==paper[0,j]:
            if co_author[1,i]==paper[1,j]:
                overlap.append(i)
        elif i < j:
            bias = j
            break

for i in range(co_author.shape[1]):
    for j in range(bias, sorted_paper.shape[1]):
        if co_author[0,i]==sorted_paper[0,j]:
            if co_author[1,i]==sorted_paper[1,j]:
                overlap.append(i)
        elif i<j:
            bias = j
            break

print('overlap ratio:',len(overlap))

new_graph = []
for i in range(co_author.shape[1]):
    if i not in overlap:
        new_graph.append([co_author[0,i],co_author[1,i]])
for i in range(paper.shape[1]):
    new_graph.append([paper[0, i], paper[1, i]])

new_graph = np.array(new_graph).T
new_graph = new_graph[:, new_graph[0, :].argsort()]
np.save(f'{dataset.dir}/fused_graph.npy',new_graph)