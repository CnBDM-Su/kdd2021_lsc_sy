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
for i in range(co_author.shape[1]):
    for j in paper