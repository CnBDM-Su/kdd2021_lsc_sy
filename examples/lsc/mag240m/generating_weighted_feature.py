import os.path as osp
from tqdm import tqdm

import time
import numpy as np
import torch

from torch_sparse import SparseTensor
from sklearn.preprocessing import MinMaxScaler
from ogb.lsc import MAG240MDataset
from root import ROOT

dataset = MAG240MDataset(ROOT)

path = f'{dataset.dir}/mini_graph/full_weighted_feat.npy'
done_flag_path = f'{dataset.dir}/mini_graph/full_weighted_feat_done.txt'
if not osp.exists(done_flag_path):  # Will take ~3 hours...
    t = time.perf_counter()
    print('Generating mini full weighted feature matrix...')
    num_dict = np.load(f'{dataset.dir}/mini_graph/num_dict.npy',allow_pickle=True).item()

    node_chunk_size = 100000
    dim_chunk_size = 64
    N = (num_dict['paper'] + num_dict['author'] +
         num_dict['institution'])

    paper_feat = np.load(f'{dataset.dir}/mini_graph/full_feat.npy')
    paper_feat = paper_feat[:num_dict['paper'],:]

    print(paper_feat.shape)

    path = f'{dataset.dir}/mini_graph/weighted_author_paper_edge.npy'
    if not osp.exists(path):
        edge_index = np.load(f'{dataset.dir}/mini_graph/author_paper_edge.npy')
        year = np.load(f'{dataset.dir}/mini_graph/paper_year.npy')
        mm = MinMaxScaler((0.5,1))
        bias = 0
        val = []
        for i in tqdm(range(num_dict['author'])):
            tmp = []
            for j in range(bias, edge_index.shape[1]):
                if i == edge_index[0,j]:
                   tmp.append(edge_index[1,j])
                if i < edge_index[0,j]:
                    bias = j
                    break
            if len(tmp)!=0:
                tmp = mm.fit_transform(year[tmp].reshape(-1,1)).ravel()
                for weight in tmp:
                    val.append(weight)
        weighted_edge = np.concatenate([edge_index,np.array(val).reshape(1,-1)],0)
        np.save(path,weighted_edge)

    row, col = torch.from_numpy(weighted_edge)
    adj_t = SparseTensor(
        row=row, col=col,val=val,
        sparse_sizes=(num_dict['author'], num_dict['paper']),
        is_sorted=True)

    # Processing 64-dim subfeatures at a time for memory efficiency.
    print('Generating author features...')

    inputs = torch.from_numpy(paper_feat)
    outputs = adj_t.matmul(inputs, reduce='mean').numpy()

    edge_index = np.load(f'{dataset.dir}/mini_graph/author_institution_edge.npy')
    row, col = torch.from_numpy(edge_index)
    adj_t = SparseTensor(
        row=col, col=row,
        sparse_sizes=(dataset.num_institutions, dataset.num_authors),
        is_sorted=False)

    print('Generating institution features...')
    # Processing 64-dim subfeatures at a time for memory efficiency.

    inputs = torch.from_numpy(paper_feat)
    outputs_2 = adj_t.matmul(inputs, reduce='mean').numpy()

    outputs = np.concatenate([paper_feat,outputs,outputs_2],0)
    np.save(f'{dataset.dir}/mini_graph/full_weighted_feat.npy',outputs)
    print(f'Done! [{time.perf_counter() - t:.2f}s]')

    with open(done_flag_path, 'w') as f:
        f.write('done')