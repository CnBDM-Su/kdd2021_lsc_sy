import time
import argparse
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn import ModuleList, Linear, BatchNorm1d, Identity
import os.path as osp
from ogb.lsc import MAG240MDataset
from root import ROOT
import numpy as np
from ogb.utils.url import makedirs

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
        assert 'y_pred' in input_dict
        y_pred = input_dict['y_pred']
        y_pred_valid = input_dict['y_pred_valid']
        # assert y_pred.shape == (146818, )

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
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--parallel', type=bool, default=False)
    parser.add_argument('--hidden_channels', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=2),
    parser.add_argument('--no_batch_norm', action='store_true')
    parser.add_argument('--relu_last', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=380000)
    parser.add_argument('--epochs', type=int, default=1000)
    args = parser.parse_args()
    print(args)

    torch.manual_seed(12345)
    gpus = [0,1,2,3,4,5,6,7]
    if torch.cuda.is_available():
        device = f'cuda:{args.device}'
    else:
        device = 'cpu'
        print('cpu')

    sup = np.load(ROOT+'/mag240m_kddcup2021/sup.npy')
    dataset = MAG240MDataset(ROOT)
    evaluator = MAG240MEvaluator()

    train_idx = dataset.get_idx_split('train')
    valid_idx = dataset.get_idx_split('valid')
    test_idx = dataset.get_idx_split('test')
    label_idx = np.concatenate([train_idx,valid_idx,test_idx],0)


    # rand = random.randint(0, 98)
    # print('random range seed:',rand)
    # no_idx = np.array(list(set(np.arange(rand*121751666//100,(rand+1)*121751666//100).tolist()) - set(label_idx.tolist())))

    t = time.perf_counter()
    print('Reading training node features...', end=' ', flush=True)
    x_train_ = torch.from_numpy(dataset.paper_feat[train_idx]).to(torch.float)
    x_train = x_train_.to(device)
    print(f'Done! [{time.perf_counter() - t:.2f}s]')
    t = time.perf_counter()
    print('Reading validation node features...', end=' ', flush=True)
    x_valid_ = torch.from_numpy(dataset.paper_feat[valid_idx]).to(torch.float)
    x_valid = x_valid_.to(device)
    print(f'Done! [{time.perf_counter() - t:.2f}s]')
    t = time.perf_counter()
    print('Reading test node features...', end=' ', flush=True)
    x_test_ = torch.from_numpy(dataset.paper_feat[test_idx]).to(torch.float)
    x_test = x_test_.to(device)
    print(f'Done! [{time.perf_counter() - t:.2f}s]')


    y_train_ = torch.from_numpy(dataset.paper_label[train_idx])
    y_train = y_train_.to(device, torch.long)
    y_valid_ = torch.from_numpy(dataset.paper_label[valid_idx])
    y_valid = y_valid_.to(device, torch.long)

    model = MLP(dataset.num_paper_features, args.hidden_channels,
                dataset.num_classes, args.num_layers, args.dropout,
                not args.no_batch_norm, args.relu_last).to(device)

    if args.parallel == True:
        model = torch.nn.DataParallel(model, device_ids=gpus)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    num_params = sum([p.numel() for p in model.parameters()])
    print(f'#Params: {num_params}')

    best_valid_acc = 0
    # for epoch in range(1, args.epochs + 1):
    #     loss = train(model, x_train, y_train, args.batch_size, optimizer)
    #     train_acc = test(model, x_train, y_train, evaluator)
    #     valid_acc = test(model, x_valid, y_valid, evaluator)
    #     if valid_acc > best_valid_acc:
    #         best_valid_acc = valid_acc
    #         with torch.no_grad():
    #             model.eval()
    #             res = {'y_pred': model(x_test).argmax(dim=-1),'y_pred_valid': model(x_valid).argmax(dim=-1)}
    #             evaluator.save_test_submission(res, 'results/mlp')
    #     if epoch % 1 == 0:
    #         print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
    #               f'Train: {train_acc:.4f}, Valid: {valid_acc:.4f}, '
    #               f'Best: {best_valid_acc:.4f}')

    for epoch in range(1, args.epochs + 1):
        loss = train(model, x_train, y_train, args.batch_size, optimizer)
        train_acc = test(model, x_train, y_train, evaluator)
        valid_acc = test(model, x_valid, y_valid, evaluator)
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            with torch.no_grad():
                model.eval()
                res = {'y_pred': model(x_test).argmax(dim=-1),'y_pred_valid': model(x_valid).argmax(dim=-1)}
                evaluator.save_test_submission(res, 'results/data_aug')
        if epoch % 1 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
                  f'Train: {train_acc:.4f}, Valid: {valid_acc:.4f}, '
                  f'Best: {best_valid_acc:.4f}')

    print('-------------second round training starts-------------')
    del x_train
    del x_valid
    del x_test
    t = time.perf_counter()
    print('Reading no label node features...', end=' ', flush=True)
    predict = None
    predict_prob = None
    for rand in range(119):
        print(rand)
        no_idx = np.array(
            list(set(np.arange(rand * 121751666 // 120, (rand + 1) * 121751666 // 120).tolist()) - set(label_idx.tolist())))
        x_no = dataset.paper_feat[no_idx]
        x_no = torch.from_numpy(x_no).to(torch.float).cpu()

        predict_ = model(x_no).argmax(dim=-1).cpu()
        if predict == None:
            predict = predict_.cpu()
        else:
            predict = torch.cat([predict,predict_],0).cpu()
        predict_prob_ = F.softmax(model(x_no),dim=1).cpu()
        del x_no
        if predict_prob == None:
            predict_prob = predict_prob_.cpu()
        else:
            predict_prob = torch.cat([predict_prob,predict_prob_],0).cpu()
        del predict_
        del predict_prob_
        record = []
        for i in sup[:, 0]:
            rank = predict_prob[predict == i, i].cpu()
            rank = rank[rank > 0.9].cpu()
            if rank.shape[0]>=sup[sup[:, 0] == i, 1][0]:
                record.append(1)
            else:
                record.append(0)

        torch.cuda.empty_cache()
        if sum(record) == len(record):
            break
    print(f'Done! [{time.perf_counter() - t:.2f}s]')


    sup_train_x_total = None
    for i in sup[:, 0]:
        # rank = predict_prob[predict == i, i]
        # rank = rank[rank > 0.9]
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
    print('old:',x_train.shape, y_train.shape)
    del x_train
    del y_train
    print('Reading training node features...', end=' ', flush=True)
    x_train_ = torch.from_numpy(dataset.paper_feat[train_idx]).to(torch.float)
    x_train = x_train_.to(device)
    print(f'Done! [{time.perf_counter() - t:.2f}s]')
    t = time.perf_counter()
    print('Reading validation node features...', end=' ', flush=True)
    x_valid_ = torch.from_numpy(dataset.paper_feat[valid_idx]).to(torch.float)
    x_valid = x_valid_.to(device)
    print(f'Done! [{time.perf_counter() - t:.2f}s]')
    t = time.perf_counter()
    print('Reading test node features...', end=' ', flush=True)
    x_test_ = torch.from_numpy(dataset.paper_feat[test_idx]).to(torch.float)
    x_test = x_test_.to(device)
    print(f'Done! [{time.perf_counter() - t:.2f}s]')
    x_train = torch.cat([x_train_, sup_train_x_total], 0).to(torch.float).to(device)
    y_train = torch.cat([y_train_, sup_train_y_total.squeeze()], 0).to(torch.long).to(device)
    print('new:',x_train.shape, y_train.shape)

    del model
    model = MLP(dataset.num_paper_features, args.hidden_channels,
                dataset.num_classes, args.num_layers, args.dropout,
                not args.no_batch_norm, args.relu_last).to(device)

    if args.parallel == True:
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
            with torch.no_grad():
                model.eval()
                res = {'y_pred': model(x_test).argmax(dim=-1),'y_pred_valid': model(x_valid).argmax(dim=-1)}
                evaluator.save_test_submission(res, 'results/data_aug')
        if epoch % 1 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
                  f'Train: {train_acc:.4f}, Valid: {valid_acc:.4f}, '
                  f'Best: {best_valid_acc:.4f}')
