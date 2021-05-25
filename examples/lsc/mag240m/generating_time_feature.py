import os.path as osp
from tqdm import tqdm

import time
import numpy as np
import torch

from torch_sparse import SparseTensor
from sklearn.preprocessing import MinMaxScaler
import sys
sys.path.append('/var/ogb/ogb/lsc')
from mag240m_mini_graph import MAG240MMINIDataset
from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity
from root import ROOT
from joblib import Parallel, delayed

dataset = MAG240MMINIDataset(ROOT)

path = f'{dataset.dir}/full_time_feat.npy'
if not osp.exists(path):  # Will take ~3 hours...
    t = time.perf_counter()
    print('Generating mini full time feature matrix...')

    node_chunk_size = 100000
    dim_chunk_size = 64
    N = (dataset.num_papers + dataset.num_authors +
         dataset.num_institutions)

    paper_feat = dataset.all_paper_feat
    print(paper_feat.shape)


    path =
    adj_t = SparseTensor(
        row=row.long(), col=col.long(),value=val.float(),
        sparse_sizes=(dataset.num_authors, dataset.num_papers),
        is_sorted=True)

    # Processing 64-dim subfeatures at a time for memory efficiency.
    print('Generating author features...')

    inputs = torch.from_numpy(paper_feat).float()
    outputs = adj_t.matmul(inputs, reduce='mean').numpy()

    edge_index = dataset.edge_index('author', 'affiliated_with', 'institution')
    row, col = torch.from_numpy(edge_index)
    adj_t = SparseTensor(
        row=col, col=row,
        sparse_sizes=(dataset.num_institutions, dataset.num_authors),
        is_sorted=False)

    print('Generating institution features...')
    # Processing 64-dim subfeatures at a time for memory efficiency.

    inputs = torch.from_numpy(paper_feat).float()
    outputs_2 = adj_t.matmul(inputs, reduce='mean').numpy()

    outputs = np.concatenate([paper_feat,outputs,outputs_2],0)
    np.save(f'{dataset.dir}/full_weighted_feat.npy',outputs)
    print(f'Done! [{time.perf_counter() - t:.2f}s]')

    with open(done_flag_path, 'w') as f:
        f.write('done')

else:
    feat = np.load(path)

path = f'{dataset.dir}/weighted_paper_paper_edge.npy'
if not osp.exists(path):
    print('Generating mini weighted paper paper edge...')
    edge_index = dataset.edge_index('paper', 'cites', 'paper')
    year = dataset.all_paper_year

    val = []
    def rbf(i, edge_index=edge_index, feat=feat):
        return rbf_kernel(feat[edge_index[0, i]].reshape(1, -1), feat[edge_index[1, i]].reshape(1, -1), 0.1)
    def cosine_sim(i, edge_index=edge_index, feat=feat):
        return cosine_similarity(feat[edge_index[0, i]].reshape(1, -1), feat[edge_index[1, i]].reshape(1, -1))
    val = Parallel(n_jobs=28)(delayed(rbf)(i) for i in range(edge_index.shape[1]))

    weighted_edge = np.concatenate([edge_index, np.array(val).reshape(1, -1)], 0)
    np.save(path, weighted_edge)
else:
    weighted_edge = np.load(path)
