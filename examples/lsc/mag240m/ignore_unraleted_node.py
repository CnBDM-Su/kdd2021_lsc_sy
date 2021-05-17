import time
import argparse
from tqdm import tqdm
import torch
import numpy as np
import os.path as osp
from ogb.lsc import MAG240MDataset
from root import ROOT

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--ignore_layer_num', type=int, default=3),
    args = parser.parse_args()
    print(args)

    torch.manual_seed(12345)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    dataset = MAG240MDataset(ROOT)
    layer_num = args.ignore_layer_num

    train_idx = dataset.get_idx_split('train')
    valid_idx = dataset.get_idx_split('valid')
    test_idx = dataset.get_idx_split('test')

    meaningful_idx = np.concatenate([train_idx, valid_idx, test_idx], 0)
    meaningful_idx = np.sort(meaningful_idx)

    pp_edge = dataset.edge_index('paper', 'cites', 'paper')

    path = f'{dataset.dir}/inverse_sorted_edge.npy'
    if not osp.exists(path):
        print('Generating inverse sorted paper paper edges...')
        t = time.perf_counter()
        ipp_edge = pp_edge[:, pp_edge[1, :].argsort()]
        np.save(path, ipp_edge)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

    else:
        ipp_edge = np.load(path)

    layer_info = {}
    for i in meaningful_idx:
        layer_info[i] = 0

    print('Searching meaningful nodes...')
    t = time.perf_counter()
    for layer in range(layer_num):
        print('start searching layer:',layer+1)
        bias_1 = 0
        bias_2 = 0
        search_domain = []
        for key in layer_info.keys():
            if layer_info[key] == layer:
                search_domain.append(key)
        for i in tqdm(range(len(search_domain))):
            i = search_domain[i]
            for j in range(bias_1,pp_edge.shape[1]):
                if i==pp_edge[0,j]:
                    if pp_edge[1,j] not in layer_info.keys():
                        layer_info[pp_edge[1,j]] = layer + 1
                if i<pp_edge[0,j]:
                    bias_1 = j
                    break

            for j in range(bias_2,ipp_edge.shape[1]):
                if i==ipp_edge[1,j]:
                    if ipp_edge[0,j] not in layer_info.keys():
                        layer_info[ipp_edge[0,j]] = layer + 1
                if i<ipp_edge[1,j]:
                    bias_2 = j
                    break
    print(f'Done! [{time.perf_counter() - t:.2f}s]')

    meaningful_idx = [i for i in layer_info.keys()]
    print('meaningful nodes num:',len(meaningful_idx))

    np.save(f'{dataset.dir}/meaningful_idx.npy',np.array(meaningful_idx))


