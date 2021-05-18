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

    path = f'{dataset.dir}/meaningful_idx.npy'
    if not osp.exists(path):
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

        meaningful_idx = np.sort(meaningful_idx)
        np.save(path,meaningful_idx)
    else:
        meaningful_idx = np.load(path)

    path = f'{dataset.dir}/sorted_author_paper_edge.npy'
    if not osp.exists(path):
        print('Generating sorted author paper edges...')
        t = time.perf_counter()
        ap_edge = dataset.edge_index('author', 'writes', 'paper')
        ap_edge = ap_edge[:, ap_edge[1, :].argsort()]
        np.save(path, ap_edge)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')
    else:
        ap_edge = np.load(path)

    meaningful_idx = np.sort(meaningful_idx)
    ai_edge = dataset.edge_index('author', 'affiliated_with', 'institution')
    path = f'{dataset.dir}/meaningful_author_idx.npy'
    if not osp.exists(path):
        print('generating author meaningful index...')
        meaningful_a = []
        for i in tqdm(range(meaningful_idx.shape[0])):
            i = meaningful_idx[i]
            for j in range(bias_1,ap_edge.shape[1]):
                if i==ap_edge[1,j]:
                    if ap_edge[0,j] not in meaningful_a:
                        meaningful_a.append(ap_edge[0,j])
                if i<ap_edge[1,j]:
                    bias_1 = j
                    break
        meaningful_a = np.sort(meaningful_a)
        np.save(path,meaningful_a)
    else:
        meaningful_a = np.load(path)

    path = f'{dataset.dir}/meaningful_institution_idx.npy'
    if not osp.exists(path):
        print('generating institution meaningful index...')
        meaningful_i = []
        for i in tqdm(range(meaningful_a.shape[0])):
            i = meaningful_a[i]
            for j in range(bias_1,ai_edge.shape[1]):
                if i==ai_edge[0,j]:
                    if ai_edge[1,j] not in meaningful_i:
                        meaningful_i.append(ai_edge[1,j])
                if i<ai_edge[0,j]:
                    bias_1 = j
                    break
        meaningful_i = np.sort(meaningful_i)
        np.save(path, meaningful_i)
    else:
        meaningful_i = np.load(path)

    path = f'{dataset.dir}/mini_graph/num_dict.npy'
    if not osp.exists(path):
        num_dict = {}
        num_dict['paper'] = meaningful_idx.shape[0]
        num_dict['author'] = meaningful_a.shape[0]
        num_dict['institution'] = meaningful_i.shape[0]
    else:
        num_dict = np.load(path,allow_pickle=True).item()

    path = f'{dataset.dir}/mini_graph/full_feat.npy'
    if not osp.exists(path):
        print('generating mini graph features...')
        t = time.perf_counter()
        N = dataset.num_papers + dataset.num_authors + dataset.num_institutions
        x = np.memmap(f'{dataset.dir}/full_feat.npy', dtype=np.float16,
                           mode='r', shape=(N, 768))
        y = x[meaningful_idx]
        y1 = x[meaningful_a+num_dict['paper']]
        y2 = x[meaningful_a + num_dict['paper']+num_dict['author']]

        y = np.concatenate([y,y1,y2],0)

        np.save(path,y)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

    p_ind_dict = {}
    for i in range(meaningful_idx.shape[0]):
        p_ind_dict[meaningful_idx[i]] = i
    a_ind_dict = {}
    for i in range(meaningful_a.shape[0]):
        a_ind_dict[meaningful_a[i]] = i
    i_ind_dict = {}
    for i in range(meaningful_i.shape[0]):
        i_ind_dict[meaningful_i[i]] = i

    path = f'{dataset.dir}/mini_graph/train_idx.npy'
    if not osp.exists(path):
        train_idx_new = []
        for i in train_idx:
            train_idx_new.append(p_ind_dict[i])
        train_idx = np.sort(train_idx_new)
        np.save(path, train_idx)
    else:
        train_idx = np.load(path)

        path = f'{dataset.dir}/mini_graph/valid_idx.npy'
        if not osp.exists(path):
            valid_idx_new = []
            for i in valid_idx:
                valid_idx_new.append(p_ind_dict[i])
            valid_idx = np.sort(valid_idx_new)
            np.save(path, valid_idx)
        else:
            valid_idx = np.load(path)

        path = f'{dataset.dir}/mini_graph/test_idx.npy'
        if not osp.exists(path):
            test_idx_new = []
            for i in test_idx:
                test_idx_new.append(p_ind_dict[i])
            test_idx = np.sort(test_idx_new)
            np.save(path, test_idx)
        else:
            test_idx = np.load(path)

    path = f'{dataset.dir}/mini_graph/paper_paper_edge.npy'
    if not osp.exists(path):
        print('generating mini graph paper_paper edge...')
        pp_edge_new = []
        for i in tqdm(range(pp_edge.shape[1])):
            if (pp_edge[0,i] in p_ind_dict.keys()) and (pp_edge[1,i] in p_ind_dict.keys()):
                pp_edge_new.append(p_ind_dict[pp_edge[0,i]],[p_ind_dict[pp_edge[1,i]]])
        pp_edge = np.array(pp_edge_new).T
        np.save(path,pp_edge)
    else:
        pp_edge = np.load(path)

    path = f'{dataset.dir}/mini_graph/author_paper_edge.npy'
    if not osp.exists(path):
        print('generating mini graph author_paper edge...')
        ap_edge_new = []
        for i in tqdm(range(ap_edge.shape[1])):
            if ap_edge[1,i] in p_ind_dict.keys():
                ap_edge.append(a_ind_dict[ap_edge[0,i]],[p_ind_dict[ap_edge[1,i]]])
        ap_edge = np.array(ap_edge_new).T
        np.save(path,ap_edge)
    else:
        ap_edge = np.load(path)

    path = f'{dataset.dir}/mini_graph/author_institution_edge.npy'
    if not osp.exists(path):
        print('generating mini graph author_institution edge...')
        ai_edge_new = []
        for i in tqdm(range(ai_edge.shape[1])):
            if ai_edge[0,i] in a_ind_dict.keys():
                ai_edge.append(a_ind_dict[ai_edge[0,i]],[i_ind_dict[ai_edge[1,i]]])
        ai_edge = np.array(ai_edge_new).T
        np.save(path,ai_edge)
    else:
        ai_edge = np.load(path)









