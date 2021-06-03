import time
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn import ModuleList, Linear, BatchNorm1d, Identity
import os.path as osp
# from ogb.lsc import MAG240MDataset
from root import ROOT
import numpy as np
import sys
sys.path.append('/var/ogb/ogb/lsc')
from mag240m_mini_graph import MAG240MMINIDataset
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
        loss = F.cross_entropy(model(x_train[idx].to(device)), y_train[idx])
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
        # assert 'y_pred' in input_dict
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
    parser.add_argument('--evaluate', type=int, default=0)
    args = parser.parse_args()
    print(args)

    torch.manual_seed(12345)
    gpus = [4,5,6,7]
    if torch.cuda.is_available():
        device = f'cuda:{args.device}'
    else:
        device = 'cpu'
        print('cpu')


    # dataset = MAG240MDataset(ROOT)
    dataset = MAG240MMINIDataset(ROOT)
    evaluator = MAG240MEvaluator()


    train_idx = dataset.get_idx_split('train')
    # train_idx = np.load(f'{dataset.dir}/new_train_idx.npy')
    valid_idx = dataset.get_idx_split('valid')
    test_idx = dataset.get_idx_split('test')

    t = time.perf_counter()
    print('Reading training node features...', end=' ', flush=True)
    x_train = dataset.paper_feat[train_idx]
    x_train = torch.from_numpy(x_train).to(torch.float).to('cpu')
    print(f'Done! [{time.perf_counter() - t:.2f}s]')
    t = time.perf_counter()
    print('Reading validation node features...', end=' ', flush=True)
    x_valid = dataset.paper_feat[valid_idx]
    x_valid = torch.from_numpy(x_valid).to(torch.float).to(device)
    print(f'Done! [{time.perf_counter() - t:.2f}s]')
    t = time.perf_counter()
    print('Reading test node features...', end=' ', flush=True)
    x_test = dataset.paper_feat[test_idx]
    x_test = torch.from_numpy(x_test).to(torch.float).to(device)
    print(f'Done! [{time.perf_counter() - t:.2f}s]')

    # label = np.load(f'{dataset.dir}/new_paper_label.npy')
    label = dataset.all_paper_label
    # y_train = torch.from_numpy(dataset.paper_label[train_idx])
    y_train = torch.from_numpy(label[train_idx])
    y_train = y_train.to(device, torch.long)
    # y_valid = torch.from_numpy(dataset.paper_label[valid_idx])
    y_valid = torch.from_numpy(label[valid_idx])
    y_valid = y_valid.to(device, torch.long)
    print(args.evaluate)
    if args.evaluate ==0:

        model = MLP(dataset.num_paper_features, args.hidden_channels,
                    dataset.num_classes, args.num_layers, args.dropout,
                    not args.no_batch_norm, args.relu_last).to(device)

        if args.parallel == True:
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
                with torch.no_grad():
                    model.eval()
                    # res = {'y_pred': model(x_test).argmax(dim=-1),'y_pred_valid': model(x_valid).argmax(dim=-1)}
                    # evaluator.save_test_submission(res, 'results/mlp')
            if epoch % 1 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
                      f'Train: {train_acc:.4f}, Valid: {valid_acc:.4f}, '
                      f'Best: {best_valid_acc:.4f}')

        # 保存
        torch.save(model.state_dict(), 'results/mlp/model.pkl')
    else:
        model = MLP(dataset.num_paper_features, args.hidden_channels,
                    dataset.num_classes, args.num_layers, args.dropout,
                    not args.no_batch_norm, args.relu_last).to(device)
        if args.parallel == True:
            model = torch.nn.DataParallel(model, device_ids=gpus)
        model.load_state_dict(torch.load('results/mlp/model.pkl'))
#___________________predict______________________________
        # feat = dataset.paper_feat
        # w = torch.t(model.state_dict()['module.lins.0.weight'])
        # bias = model.state_dict()['module.lins.0.bias']
        # batch_size = 600000
        # con = []
        # for i in range(feat.shape[0]//600000+1):
        #     end = min((i+1)*batch_size,feat.shape[0])
        #     feat1 = torch.from_numpy(feat[i*batch_size:end]).to(device).to(torch.half)
        #     con.append(torch.matmul(feat1,w.to(torch.half))+bias.to(torch.half))
        #
        # con = torch.cat(con).numpy()
        # np.save(f'{dataset.dir}/256dim/node_feat.npy')
#____________________test___________________________
        y_relate = np.load(f'{dataset.dir}/data_rule_result_relate.npy')
        y_rule = np.load(f'{dataset.dir}/data_rule_result.npy')[y_relate]
        y_relate_true = torch.from_numpy(label[y_relate])
        x_relate = torch.from_numpy(dataset.paper_feat[y_relate]).to(torch.float).to('cpu')
        y_mlp = model(x_relate).argmax(dim=-1).cpu().numpy()
        a = set(np.where(y_rule != y_relate_true.cpu().numpy())[0])
        b = set(np.where(y_mlp == y_relate_true.cpu().numpy())[0])
        c = set(np.where(y_rule == y_relate_true.cpu().numpy())[0])
        d = set(np.where(y_mlp != y_relate_true.cpu().numpy())[0])
        print('rule_right',len(c))
        print('rule_wrong:',len(a))
        print('mlp_right:', len(b))
        print('mlp_wrong:', len(d))
        mlp_easy = list(a & b)
        mlp_hard = list(c & d)
        print('easy:',len(mlp_easy))
        print('hard:', len(mlp_hard))
        mlp_hard = mlp_hard[:len(mlp_easy)]
        x_easy = dataset.paper_feat[mlp_easy]
        x_hard = dataset.paper_feat[mlp_hard]
        w = torch.t(model.state_dict()['module.lins.0.weight'])
        # x_easy = torch.matmul(torch.from_numpy(x_easy).cpu().to(torch.half),w.cpu().to(torch.half)) + model.state_dict()['module.lins.0.bias'].cpu().to(torch.half)
        # x_hard = torch.matmul(torch.from_numpy(x_hard).cpu().to(torch.half),w.cpu().to(torch.half)) + model.state_dict()['module.lins.0.bias'].cpu().to(torch.half)
        print(x_easy.shape)
        print(x_hard.shape)
        # x_easy = x_easy.numpy()
        # x_hard = x_hard.numpy()
        label_easy = dataset.all_paper_label[mlp_easy]
        label_hard = dataset.all_paper_label[mlp_hard]

        from sklearn.metrics.pairwise import cosine_distances
        easy_in_dis = []
        center_easy = []
        hard_in_dis = []
        center_hard = []
        easy_type = []
        hard_type = []
        for i in range(153):
            if x_easy[label_easy==i].shape[0] != 0:
                easy_type.append(i)
                # easy_in_dis.append(np.mean(cosine_distances(x_easy[label_easy==i])))
                center_easy.append(np.mean(x_easy[label_easy==i],0))

        for i in range(153):
            if x_hard[label_hard==i].shape[0] != 0:
                hard_type.append(i)
                # hard_in_dis.append(np.mean(cosine_distances(x_hard[label_hard==i])))
                center_hard.append(np.mean(x_hard[label_hard==i],0))


        easy_in_dis = []
        easy_out_dis = []
        for i in range(len(easy_type)):
            label = easy_type[i]
            for j in x_easy[label_easy == label]:
                easy_in_dis.append(cosine_distances([j,center_easy[i]])[0,1])
                easy_out_dis_tmp = []
                for k in range(len(easy_type)):
                    if k != i:
                        easy_out_dis_tmp.append(cosine_distances([j, center_easy[k]])[0, 1])
                easy_out_dis.append(np.mean(easy_out_dis_tmp))

        S_easy = []
        for i in range(len(easy_in_dis)):
            S_easy.append((easy_out_dis[i] - easy_in_dis[i]) / max(easy_out_dis[i], easy_in_dis[i]))
        S_easy=np.mean(S_easy)

        hard_in_dis = []
        hard_out_dis = []
        for i in range(len(hard_type)):
            label = hard_type[i]
            for j in x_hard[label_hard == label]:
                hard_in_dis.append(cosine_distances([j,center_hard[i]])[0,1])
                hard_out_dis_tmp = []
                for k in range(len(hard_type)):
                    if k != i:
                        hard_out_dis_tmp.append(cosine_distances([j, center_hard[k]])[0, 1])
                hard_out_dis.append(np.mean(hard_out_dis_tmp))

        S_hard = []
        for i in range(len(hard_in_dis)):
            S_hard.append((hard_out_dis[i] - hard_in_dis[i]) / max(hard_out_dis[i], hard_in_dis[i]))
        S_hard=np.mean(S_hard)


        # easy_in_dis = np.mean(easy_in_dis)
        # hard_in_dis = np.mean(hard_in_dis)
        # easy_out_dis = np.mean(cosine_distances(center_easy))
        # hard_out_dis = np.mean(cosine_distances(center_hard))
        # S_easy = (easy_out_dis - easy_in_dis) / max(easy_out_dis, easy_in_dis)
        # S_hard = (hard_out_dis - hard_in_dis) / max(hard_out_dis, hard_in_dis)
        print('easy Silhouette Coefficient:',S_easy)
        print('hard Silhouette Coefficient:', S_hard)


