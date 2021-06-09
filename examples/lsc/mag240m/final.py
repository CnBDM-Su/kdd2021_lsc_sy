import time
import argparse
from tqdm import tqdm
import torch
import os.path as osp
from root import ROOT
import numpy as np
import sys
sys.path.append('/var/ogb/ogb/lsc')
from mag240m_mini_graph import MAG240MMINIDataset
from ogb.utils.url import makedirs
from copy import deepcopy
import glob
from final_cs import kdd_cs
from final_mlp import kdd_mlp
from final_rgat import kdd_rgat

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
        assert y_pred.shape == (146818, )

        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        y_pred = y_pred.astype(np.short)

        makedirs(dir_path)
        filename = osp.join(dir_path, 'y_pred_mag240m')
        np.savez_compressed(filename, y_pred=y_pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--step', type=int, default=0)

    #mlp_parameter
    parser.add_argument('--mlp_hidden_channels', type=int, default=256)
    parser.add_argument('--mlp_num_layers', type=int, default=2),
    parser.add_argument('--mlp_no_batch_norm', action='store_true')
    parser.add_argument('--mlp_relu_last', action='store_true')
    parser.add_argument('--mlp_dropout', type=float, default=0.5)
    parser.add_argument('--mlp_lr', type=float, default=0.01)
    parser.add_argument('--mlp_batch_size', type=int, default=380000)
    parser.add_argument('--mlp_epochs', type=int, default=1000)

    # rgat_parameter
    parser.add_argument('--rgat_hidden_channels', type=int, default=1024)
    parser.add_argument('--rgat_batch_size', type=int, default=1024)
    parser.add_argument('--rgat_dropout', type=float, default=0.5)
    parser.add_argument('--rgat_epochs', type=int, default=100)
    parser.add_argument('--rgat_model', type=str, default='rgat',
                        choices=['rgat', 'rgraphsage'])
    parser.add_argument('--rgat_sizes', type=str, default='25-15')
    parser.add_argument('--rgat_resume', type=int, default=None)

    # cs_parameter
    parser.add_argument('--num_correction_layers', type=int, default=3)
    parser.add_argument('--correction_alpha', type=float, default=1.0)
    parser.add_argument('--num_smoothing_layers', type=int, default=2)
    parser.add_argument('--smoothing_alpha', type=float, default=0.2)


    args = parser.parse_args()
    args.rgat_sizes = [int(i) for i in args.rgat_sizes.split('-')]
    print(args)

    gpus = [4,5,6,7]
    if torch.cuda.is_available():
        device = f'cuda:{args.device}'
    else:
        device = 'cpu'
        print('cpu')

    dataset = MAG240MMINIDataset(ROOT)
    evaluator = MAG240MEvaluator()

    train_idx = dataset.get_idx_split('train')
    valid_idx = dataset.get_idx_split('valid')
    test_idx = dataset.get_idx_split('test')

    train_idx = np.concatenate([train_idx,valid_idx],0)

    x = np.load(f'{dataset.dir}/paper_relation_weighted_feat.npy')
    print(x.shape)
    t = time.perf_counter()

    print('Reading training node features...', end=' ', flush=True)
    x_train = x[train_idx]
    x_train = torch.from_numpy(x_train).to(torch.float).to('cpu')
    print(f'Done! [{time.perf_counter() - t:.2f}s]')
    t = time.perf_counter()
    print('Reading validation node features...', end=' ', flush=True)
    x_valid = x[valid_idx]
    x_valid = torch.from_numpy(x_valid).to(torch.float).to(device)
    print(f'Done! [{time.perf_counter() - t:.2f}s]')
    t = time.perf_counter()
    print('Reading test node features...', end=' ', flush=True)
    x_test = x[test_idx]
    x_test = torch.from_numpy(x_test).to(torch.float).to(device)
    print(f'Done! [{time.perf_counter() - t:.2f}s]')

    paper_label = dataset.all_paper_label
    y_train = torch.from_numpy(paper_label[train_idx])
    y_train = y_train.to(device, torch.long)
    y_valid = torch.from_numpy(paper_label[valid_idx])
    y_valid = y_valid.to(device, torch.long)
    print(args.evaluate)

    if args.step==0:

        kdd_mlp(dataset, train_idx, valid_idx, test_idx, paper_label,
                device, args.parallel, args.mlp_hidden_channels, args.mlp_num_layers, args.mlp_no_batch_norm,
                args.mlp_relu_last, args.mlp_dropout, args.mlp_lr, args.mlp_batch_size, args.mlp_epochs)

        kdd_rgat(dataset, train_idx, valid_idx, test_idx, paper_label,
                 args.rgat_hidden_channels, args.rgat_batch_size, args.rgat_dropout, args.rgat_epochs, args.rgat_model,
                 args.rgat_sizes, device, args.rgat_resume)

        kdd_cs(dataset, train_idx, valid_idx, test_idx, paper_label,
                args.num_correction_layers, args.correction_alpha, args.num_smoothing_layers, args.smoothing_alpha)

    elif args.step==1:
        kdd_mlp(dataset, train_idx, valid_idx, test_idx, paper_label,
                device, args.parallel, args.mlp_hidden_channels, args.mlp_num_layers, args.mlp_no_batch_norm,
                args.mlp_relu_last, args.mlp_dropout, args.mlp_lr, args.mlp_batch_size, args.mlp_epochs)

    elif args.step==2:
        kdd_rgat(dataset, train_idx, valid_idx, test_idx, paper_label,
                 args.rgat_hidden_channels, args.rgat_batch_size, args.rgat_dropout, args.rgat_epochs, args.rgat_model,
                 args.rgat_sizes, device, args.rgat_resume)

    elif args.step==3:
        kdd_cs(dataset, train_idx, valid_idx, test_idx, paper_label,
               args.num_correction_layers, args.correction_alpha, args.num_smoothing_layers, args.smoothing_alpha)

