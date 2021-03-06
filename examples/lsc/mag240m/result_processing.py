import os.path as osp
import time
import argparse
from tqdm import tqdm
from copy import deepcopy
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
class MAG240MEvaluator:
    def eval(self, input_dict):
        assert 'y_pred' in input_dict and 'y_true' in input_dict

        y_pred, y_true = input_dict['y_pred'], input_dict['y_true']

        if not isinstance(y_pred, torch.Tensor):
            y_pred = torch.from_numpy(y_pred)
        if not isinstance(y_true, torch.Tensor):
            y_true = torch.from_numpy(y_true)

        assert (y_true.numel() == y_pred.numel())
        assert (y_true.dim() == y_pred.dim() == 1)

        return {'acc': int((y_true == y_pred).sum()) / y_true.numel()}

    def save_test_submission(self, input_dict, dir_path):
        # assert 'y_pred' in input_dict
        y_pred = input_dict['y_pred']
        y_pred_valid = input_dict['y_pred_valid']
        assert y_pred.shape == (146818, )

        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        y_pred = y_pred.astype(np.short)

        if isinstance(y_pred_valid, torch.Tensor):
            y_pred_valid = y_pred_valid.cpu().numpy()
        y_pred_valid = y_pred_valid.astype(np.short)

        makedirs(dir_path)
        filename = osp.join(dir_path, 'y_pred_mag240m')
        np.savez_compressed(filename, y_pred=y_pred, y_pred_valid=y_pred_valid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_correction_layers', type=int, default=3)
    parser.add_argument('--correction_alpha', type=float, default=1.0)
    parser.add_argument('--num_smoothing_layers', type=int, default=2)
    parser.add_argument('--smoothing_alpha', type=float, default=0.8)
    parser.add_argument('--mini_graph', type=bool, default=False)
    args = parser.parse_args()
    print(args)

    evaluator = MAG240MEvaluator()

    if args.mini_graph:
        dataset = MAG240MMINIDataset(ROOT)
        # save_path = 'results/mini_cs_weighted'
        save_path = 'results/rgat_cs'

    else:
        dataset = MAG240MDataset(ROOT)
        save_path = 'results/cs'

    train_idx = torch.from_numpy(dataset.get_idx_split('train'))
    valid_idx = torch.from_numpy(dataset.get_idx_split('valid'))
    test_idx = torch.from_numpy(dataset.get_idx_split('test'))
    idx = np.concatenate([dataset.get_idx_split('train'),dataset.get_idx_split('valid'),dataset.get_idx_split('test')],0)
    paper_label = dataset.paper_label

    print('Reading MLP soft prediction...', end=' ', flush=True)
    t = time.perf_counter()
    y_pred = torch.from_numpy(np.load(save_path+'/rgat_pred.npz')['y_pred'])
    # y_pred = torch.from_numpy(np.load(save_path + '/rgat_cs_pred.npy'))
    # y_pred = torch.from_numpy(np.load(save_path+'/pred.npy'))
    print(f'Done! [{time.perf_counter() - t:.2f}s]')

    t = time.perf_counter()
    print('Reading adjacency matrix...', end=' ', flush=True)
    path = f'{dataset.dir}/paper_to_paper_symmetric_gcn.pt'
    if osp.exists(path):
        adj_t = torch.load(path)
    else:
        path_sym = f'{dataset.dir}/paper_to_paper_symmetric.pt'
        if osp.exists(path_sym):
            adj_t = torch.load(path_sym)
        else:
            edge_index = dataset.edge_index('paper', 'cites', 'paper')
            edge_index = torch.from_numpy(edge_index)
            adj_t = SparseTensor(
                row=edge_index[0], col=edge_index[1],
                sparse_sizes=(dataset.num_papers, dataset.num_papers),
                is_sorted=True)
            adj_t = adj_t.to_symmetric()
            torch.save(adj_t, path_sym)
        adj_t = gcn_norm(adj_t, add_self_loops=True)
        torch.save(adj_t, path)
    print(f'Done! [{time.perf_counter() - t:.2f}s]')

    y_train = torch.from_numpy(paper_label[train_idx]).to(torch.long)
    y_valid = torch.from_numpy(paper_label[valid_idx]).to(torch.long)
    # edge_index = np.load(f'{dataset.dir}/weighted_paper_paper_edge.npy')
    # edge_index = torch.from_numpy(edge_index)
    # adj_t = adj_t.set_value(edge_index[2], layout='coo')

    model = CorrectAndSmooth(args.num_correction_layers, args.correction_alpha,
                             args.num_smoothing_layers, args.smoothing_alpha,
                             autoscale=True)

    t = time.perf_counter()
    print('Correcting predictions...', end=' ', flush=True)
    assert abs((float(y_pred.sum()) / y_pred.size(0)) - 1.0) < 1e-2

    numel = int(train_idx.sum()) if train_idx.dtype == torch.bool else train_idx.size(0)
    assert y_train.size(0) == numel

    y_pred = model.correct(y_pred, y_train, train_idx, adj_t)
    print(f'Done! [{time.perf_counter() - t:.2f}s]')

    t = time.perf_counter()
    print('Smoothing predictions...', end=' ', flush=True)
    y_pred = model.smooth(y_pred, y_train, train_idx, adj_t)
    print(f'Done! [{time.perf_counter() - t:.2f}s]')

    print('result processing...', end=' ', flush=True)
    # print(y_pred.shape)
    from sklearn.ensemble import RandomForestClassifier
    # from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import MinMaxScaler
    # from sklearn.tree import DecisionTreeClassifier
    y_correct = torch.from_numpy(np.load(f'{dataset.dir}/data_rule_result.npy')).to(torch.long)
    index_correct = np.load(f'{dataset.dir}/data_rule_result_relate.npy')
    # model_rf = DecisionTreeClassifier()
    # y_pred_ = deepcopy(y_pred).numpy()
    # new_data = np.concatenate([y_pred_[idx],y_correct],1)
    # rf_valid = new_data[train_idx.shape[0]:valid_idx.shape[0]+train_idx.shape[0]]
    # rf_test = new_data[valid_idx.shape[0]+train_idx.shape[0]:]
    # mm = MinMaxScaler()
    # new_data = mm.fit_transform(new_data)
    # rf_train_x = new_data[:train_idx.shape[0]]
    # rf_train_y = y_train.numpy()
    # model_rf.fit(rf_train_x,rf_train_y)
    # y_pred_ = torch.from_numpy(model_rf.predict_proba(new_data))

    # y_pred_ = y_pred.argmax(dim=-1)
    t = time.perf_counter()

    # correct_index = np.load(f'{dataset.dir}/changed_valid_idx.npy')
    # correct_index = np.array(list(set(correct_index) & set(modify_index)))
    # for i in correct_index:
    #     y_pred_[i] = y_correct[i]

    c = 0
    # b_lis = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    # acc_lis = []
    # def best_ratio(b, y_pred=y_pred, y_correct=y_correct):
    #     a = 1
    #     y_pred_ = y_pred.argmax(dim=-1)
    #     for i in range(valid_idx.shape[0]):
    #         ind = valid_idx[i]
    #         if y_correct[i].sum()!=0:
                # y_pred_[ind] = (a * y_pred[ind] + b * y_correct[ind])/(a+b)


    #     train_acc = evaluator.eval({
    #         'y_true': y_train,
    #         'y_pred': y_pred_.argmax(dim=-1)[:train_idx.shape[0]]
    #     })['acc']
    #     valid_acc = evaluator.eval({
    #         'y_true': y_valid,
    #         'y_pred': y_pred_.argmax(dim=-1)[train_idx.shape[0]:valid_idx.shape[0]+train_idx.shape[0]]
    #     })['acc']
    #     print(f'Train: {train_acc:.4f}, Valid: {valid_acc:.4f}')
    #
    #     return y_pred_, train_acc
    #
    # for i in b_lis:
    #     y_pred_tmp, train_acc = best_ratio(i)
    #     acc_lis.append(train_acc)
    #     if len(acc_lis)>1:
    #         if acc_lis[-1] > acc_lis[-2]:
    #             y_pred_best = y_pred_tmp
    #         else:
    #             y_pred = y_pred_best
    #             break
    y_pred_= y_pred.argmax(dim=-1)
    y_pred_[index_correct] = y_correct[index_correct]
    train_acc = evaluator.eval({
        'y_true': y_train,
        'y_pred': y_pred_[train_idx]
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_valid,
        'y_pred': y_pred_[valid_idx]
    })['acc']
    print(f'Train: {train_acc:.4f}, Valid: {valid_acc:.4f}')

    # train_acc = evaluator.eval({
    #     'y_true': y_train,
    #     'y_pred': y_pred.argmax(dim=-1)[train_idx]
    # })['acc']
    # valid_acc = evaluator.eval({
    #     'y_true': y_valid,
    #     'y_pred': y_pred.argmax(dim=-1)[valid_idx]
    # })['acc']
    # print(f'Train: {train_acc:.4f}, Valid: {valid_acc:.4f}')

    # print('correct num:',correct_index.shape[0])
    # initial_pred = y_pred.argmax(dim=-1)
    # final_pred = y_pred_.argmax(dim=-1)
    # y_true = torch.from_numpy(paper_label).to(torch.long)
    # TT = 0
    # TF = 0
    # FT = 0
    # FF = 0
    # TF_initial = []
    # TF_final = []
    # FT_initial = []
    # FT_final = []
    # for i in valid_idx:
    #     if (y_true[i] == initial_pred[i]) and (y_true[i] == final_pred[i]):
    #         TT += 1
    #     elif (y_true[i] == initial_pred[i]) and (y_true[i] != final_pred[i]):
    #         TF += 1
    #         # TF_initial.append(initial_pred[i])
    #         # TF_final.append(final_pred[i])
    #     elif (y_true[i] != initial_pred[i]) and (y_true[i] == final_pred[i]):
    #         FT += 1
    #         # FT_initial.append(initial_pred[i])
    #         # FT_final.append(final_pred[i])
    #     elif (y_true[i] != initial_pred[i]) and (y_true[i] != final_pred[i]):
    #         FF += 1
    # print('TT:', TT)
    # print('TF:', TF)
    # print('FT:', FT)
    # print('FF:', FF)
    # np.savez_compressed('confusion.npz', TF_initial=np.array(TF_initial), TF_final=np.array(TF_final), FT_initial=np.array(FT_initial), FT_final=np.array(FT_final))

    # res = {'y_pred': y_pred_[test_idx], 'y_pred_valid' : y_pred_[valid_idx]}
    # evaluator.save_test_submission(res, save_path)