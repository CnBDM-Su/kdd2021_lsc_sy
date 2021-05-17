import time
import argparse
from tqdm import tqdm

import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn import ModuleList, Linear, BatchNorm1d, Identity
import random
import os.path as osp

from ogb.utils.url import makedirs
from ogb.lsc import MAG240MDataset
from root import ROOT

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
        y_pred_valid = input_dict['y_pred_valid']
        # assert y_pred.shape == (146818, )

        if isinstance(y_pred_valid, torch.Tensor):
            y_pred_valid = y_pred_valid.cpu().numpy()
        y_pred_valid = y_pred_valid.astype(np.short)

        makedirs(dir_path)
        filename = osp.join(dir_path, 'y_pred_mag240m')
        np.savez_compressed(filename, y_pred_valid=y_pred_valid)

class MLP(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int, dropout: float = 0.0,
                 batch_norm: bool = True, relu_last: bool = False):
        super(MLP, self).__init__()

        self.lins = ModuleList()
        self.lins.append(Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.lins.append(Linear(hidden_channels, out_channels))

        self.batch_norms = ModuleList()
        for _ in range(num_layers - 1):
            norm = BatchNorm1d(hidden_channels) if batch_norm else Identity()
            self.batch_norms.append(norm)

        self.dropout = dropout
        self.relu_last = relu_last

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for batch_norm in self.batch_norms:
            batch_norm.reset_parameters()

    def forward(self, x):
        for lin, batch_norm in zip(self.lins[:-1], self.batch_norms):
            x = lin(x)
            if self.relu_last:
                x = batch_norm(x).relu_()
            else:
                x = batch_norm(x.relu_())
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


def train(model, x_train, y_train, batch_size, optimizer):
    model.train()

    total_loss = 0
    for idx in DataLoader(range(y_train.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x_train[idx]), y_train[idx])
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * idx.numel()

    return total_loss / y_train.size(0)


@torch.no_grad()
def test(model, x_eval, y_eval, evaluator):
    model.eval()
    y_pred = model(x_eval).argmax(dim=-1)
    return evaluator.eval({'y_true': y_eval, 'y_pred': y_pred})['acc']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--hidden_channels', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=2),
    parser.add_argument('--no_batch_norm', action='store_true')
    parser.add_argument('--relu_last', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=380000)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--evaluate', type=bool, default=False)
    parser.add_argument('--p_batch_size', type=int, default=40000)
    args = parser.parse_args()
    print(args)

    torch.manual_seed(12345)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    gpus = [0,1,2,3,4,5,6,7]

    dataset = MAG240MDataset(ROOT)
    evaluator = MAG240MEvaluator()

    train_idx = dataset.get_idx_split('train')
    valid_idx = dataset.get_idx_split('valid')
    test_idx = dataset.get_idx_split('test')
    label_idx = np.concatenate([train_idx, valid_idx, test_idx], 0)

    path = f'{dataset.dir}/sorted_author_paper_edge.npy'
    if not osp.exists(path):
        print('Generating sorted author paper edges...')
        t = time.perf_counter()
        ap_edge = dataset.edge_index('author', 'writes', 'paper')
        ap_edge = ap_edge[:, ap_edge[1, :].argsort()]
        np.save(path, ap_edge)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

    path = f'{dataset.dir}/paper_relation_feat.npy'
    if not osp.exists(path):
        print('Generating paper relation features...')
        t = time.perf_counter()
        N = dataset.num_papers + dataset.num_authors + dataset.num_institutions
        x = np.memmap(f'{dataset.dir}/full_feat.npy', dtype=np.float16,
                           mode='r', shape=(N, 768))
        y = np.memmap(path, dtype=np.float16, mode='w+',
                      shape=(dataset.num_papers, 1536))
        ap_edge = np.load(f'{dataset.dir}/sorted_author_paper_edge.npy')
        bias = 0
        p_batch_size = args.p_batch_size
        for p_batch in range(dataset.num_papers//p_batch_size):
            fea_ = []
            for i in tqdm(range(p_batch*p_batch_size,(p_batch+1)*p_batch_size)):
                sign = 0
                fea = []
                for j in range(bias,len(ap_edge[0])):
                    if ap_edge[1,j] == i:
                        fea.append(ap_edge[0,j]+dataset.num_papers)
                        sign = 1
                    if len(fea) > 20:
                        break
                    if (sign==1) and (ap_edge[1,j] != i):
                        break
                bias = j
                fea = x[fea]
                fea_.append(np.mean(fea, 0))
            fea_ = np.array(fea_)
            y[p_batch*p_batch_size:(p_batch+1)*p_batch_size] = np.concatenate([x[p_batch*p_batch_size:(p_batch+1)*p_batch_size],fea_],1)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

    else:
        x_fr = np.memmap(path, dtype=np.float16, mode='r',
                      shape=(dataset.num_papers, 1536))

    if args.evaluate==False:
        t = time.perf_counter()

        print('Reading training node features...', end=' ', flush=True)
        # x_train = dataset.paper_feat[train_idx]
        x_train_ = torch.from_numpy(x_fr[train_idx]).to(torch.float)
        x_train = x_train_.to(device)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')
        t = time.perf_counter()
        print('Reading validation node features...', end=' ', flush=True)
        x_valid_ = torch.from_numpy(x_fr[valid_idx]).to(torch.float)
        x_valid = x_valid_.to(device)
        # x_valid = dataset.paper_feat[valid_idx]
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

        y_train_ = torch.from_numpy(dataset.paper_label[train_idx]).to(torch.long)
        y_train = y_train_.to(device)
        y_valid_ = torch.from_numpy(dataset.paper_label[valid_idx]).to(torch.long)
        y_valid = y_valid_.to(device)

        makedirs('results/cs')
        model = MLP(dataset.num_paper_features*2, args.hidden_channels,
                    dataset.num_classes, args.num_layers, args.dropout,
                    not args.no_batch_norm, args.relu_last).to(device)
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model, device_ids=gpus)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        num_params = sum([p.numel() for p in model.parameters()])
        print(f'#Params: {num_params}')

        best_valid_acc = 0
        for epoch in range(1, args.epochs + 1):
            loss = train(model, x_train, y_train, args.batch_size, optimizer)
            train_acc = test(model, x_train, y_train, evaluator)
            valid_acc = test(model, x_valid, y_valid, evaluator)
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                torch.save(model.state_dict(), 'results/cs/model.pt')
                with torch.no_grad():
                    model.eval()
                    res = {'y_pred_valid': model(x_valid).argmax(dim=-1)}
                    evaluator.save_test_submission(res, 'results/cs')
            if epoch % 100 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
                      f'Train: {train_acc:.4f}, Valid: {valid_acc:.4f}, '
                      f'Best: {best_valid_acc:.4f}')

        print('-------------error analysis-------------')

        predval = np.load('results/cs/y_pred_mag240m.npz')['y_pred_valid']

        tr_label = dataset.all_paper_label[train_idx]
        val_label = dataset.all_paper_label[valid_idx]

        err_count = {}
        for i in range(val_label.shape[0]):
            if val_label[i] != predval[i]:
                if val_label[i] not in err_count.keys():
                    err_count[val_label[i]] = 1
                else:
                    err_count[val_label[i]] += 1

        val_count = {}
        for i in val_label:
            if i not in val_count.keys():
                val_count[i] = 1
            else:
                val_count[i] += 1

        tr_count = {}
        for i in tr_label:
            if i not in tr_count.keys():
                tr_count[i] = 1
            else:
                tr_count[i] += 1

        err_rate = {}
        for i in val_count.keys():
            err_rate[i] = err_count[i]/val_count[i]

        # tr_count = tr_label.label.value_counts().sort_index()
        # tr_ratio = tr_label.label.value_counts().sort_index() / tr_label.shape[0]
        # print('ratio mean:', np.array(tr_ratio).mean())
        # print('count mean:', np.array(tr_count).mean())

        sup = {}
        for i in err_rate.keys():
            if (err_rate[i]<=0.3) and (tr_count[i] <20000):
                sup[i] = 20000 - tr_count[i]
        sup = np.array([[i for i in sup.keys()], [i for i in sup.values()]]).T
        sup = sup.astype(int)
        print(sup)

        print('-------------second round training starts-------------')
        t = time.perf_counter()
        print('Reading no label node features...', end=' ', flush=True)
        predict = None
        predict_prob = None
        sup_train_x_total = None
        finish_record = {}
        for rand in range(1):
            no_idx = np.array(
                list(set(np.arange(rand * 121751666 // 100, (rand + 1) * 121751666 // 100).tolist()) - set(
                    label_idx.tolist())))
            x_no = x_fr[no_idx]
            x_no = torch.from_numpy(x_no).to(torch.float)
            with torch.no_grad():
                predict = model(x_no).argmax(dim=-1)
                predict_prob = F.softmax(model(x_no), dim=1)

            accu_list = []
            for i in sup[:, 0]:
                rank = predict_prob[predict == i, i]
                rank = rank[rank > 0.9]
                accu_list.append(rank.shape[0])
                fill_num = min(rank.shape[0], sup[sup[:, 0] == i, 1][0])
                ind = torch.sort(rank, descending=True).indices[:fill_num]
                sup_train_x = x_no[predict == i, :][ind]
                sup_train_y = torch.ones(fill_num).reshape(-1, 1) * i
                if sup_train_x_total == None:
                    sup_train_x_total = sup_train_x
                    sup_train_y_total = sup_train_y
                else:
                    sup_train_x_total = torch.cat([sup_train_x_total, sup_train_x], 0)
                    sup_train_y_total = torch.cat([sup_train_y_total, sup_train_y], 0)

                    if rank.shape[0] >= sup[sup[:, 0] == i, 1][0]:
                        if i not in finish_record.keys():
                            finish_record[i] = rank.shape[0]
                        else:
                            finish_record[i] += rank.shape[0]

            del predict
            del predict_prob
            sup[:, -1] = sup[:, -1] - accu_list
            sup = sup[sup[:, -1] > 0]

            torch.cuda.empty_cache()
            print(sup.shape[0])
            if sup.shape[0] == 0:
                print('apply {} batchs for data augmentation'.format(rand + 1))
                print('finished class contains:', finish_record)
                break
        print('apply {} batchs for data augmentation'.format(rand + 1))
        print('finished class contains:', finish_record)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

        print('old:', x_train.shape, y_train.shape)
        del x_train
        del y_train
        x_train = torch.cat([x_train_, sup_train_x_total], 0).to(torch.float).to(device)
        y_train = torch.cat([y_train_, sup_train_y_total.squeeze()], 0).to(torch.long).to(device)
        print('new:', x_train.shape, y_train.shape)

        del model
        model = MLP(dataset.num_paper_features*2, args.hidden_channels,
                    dataset.num_classes, args.num_layers, args.dropout,
                    not args.no_batch_norm, args.relu_last).to(device)

        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model, device_ids=gpus)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        num_params = sum([p.numel() for p in model.parameters()])
        print(f'#Params: {num_params}')

        for epoch in range(1, args.epochs + 1):
            loss = train(model, x_train, y_train, args.batch_size, optimizer)
            train_acc = test(model, x_train, y_train, evaluator)
            valid_acc = test(model, x_valid, y_valid, evaluator)
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                torch.save(model.state_dict(), 'results/cs/model.pt')
            if epoch % 100 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
                      f'Train: {train_acc:.4f}, Valid: {valid_acc:.4f}, '
                      f'Best: {best_valid_acc:.4f}')
    else:
        model = MLP(dataset.num_paper_features, args.hidden_channels,
                    dataset.num_classes, args.num_layers, args.dropout,
                    not args.no_batch_norm, args.relu_last).to(device)
    model.load_state_dict(torch.load('results/cs/model.pt'))
    model.eval()

    pbar = tqdm(total=dataset.num_papers)
    pbar.set_description('Saving model predictions')

    out = []
    for i in range(0, dataset.num_papers, args.batch_size):
        x = x_fr[i:min(i + args.batch_size, dataset.num_papers)]
        x = torch.from_numpy(x).to(torch.float).to(device)
        with torch.no_grad():
            out.append(model(x).softmax(dim=-1).cpu().numpy())
        pbar.update(x.size(0))
    pbar.close()
    np.save('results/cs/pred.npy', np.concatenate(out, axis=0))
    # np.savez('results/cs/pred.npz', *out)

