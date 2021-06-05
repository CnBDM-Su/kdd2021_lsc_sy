from pl_bolts.models.self_supervised import Moco_v2
import os
import time
import glob
import argparse
import os.path as osp
from tqdm import tqdm
import sys
from typing import Optional, List, NamedTuple

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Sequential, Linear, BatchNorm1d, ReLU, Dropout
from torch.optim.lr_scheduler import StepLR

from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import (LightningDataModule, LightningModule, Trainer,
                               seed_everything)

from torch_sparse import SparseTensor
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.data import NeighborSampler

from ogb.lsc import MAG240MDataset
from root import ROOT
from ogb.utils.url import makedirs
import sys
sys.path.append('/var/ogb/ogb/lsc')
from mag240m_mini_graph import MAG240MMINIDataset

class Batch(NamedTuple):
    x: Tensor
    y: Tensor
    adjs_t: List[SparseTensor]

    def to(self, *args, **kwargs):
        return Batch(
            x=self.x.to(*args, **kwargs),
            y=self.y.to(*args, **kwargs),
            adjs_t=[adj_t.to(*args, **kwargs) for adj_t in self.adjs_t],
        )


class MAG240M(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, sizes: List[int], mini: bool):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.sizes = sizes
        self.mini = mini
        self.setup()

    @property
    def num_features(self) -> int:
        return 768

    @property
    def num_classes(self) -> int:
        return 153

    @property
    def num_relations(self) -> int:
        return 5

    def setup(self, stage: Optional[str] = None):
        t = time.perf_counter()
        print('Reading dataset...', end=' ', flush=True)
        if self.mini:
            dataset = MAG240MMINIDataset(self.data_dir)
        else:
            dataset = MAG240MDataset(self.data_dir)

        self.train_idx = torch.from_numpy(dataset.get_idx_split('train'))
        self.train_idx = self.train_idx
        self.train_idx.share_memory_()
        self.val_idx = torch.from_numpy(dataset.get_idx_split('valid'))
        self.val_idx.share_memory_()
        self.test_idx = torch.from_numpy(dataset.get_idx_split('test'))
        self.test_idx.share_memory_()

        N = dataset.num_papers + dataset.num_authors + dataset.num_institutions

        self.x = np.load(f'{dataset.dir}/full_feat.npy')
        self.y = torch.from_numpy(dataset.all_paper_label)
        self.file_batch_size = N//1000

        path = f'{dataset.dir}/full_adj_t.pt'
        self.adj_t = torch.load(path)
        print(f'Done! [{time.perf_counter() - t:.2f}s]')

    def train_dataloader(self):
        ns = NeighborSampler(self.adj_t, node_idx=self.train_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size, shuffle=True,
                               num_workers=16)
        return ns

    def all_dataloader(self):
        if self.mini:
            dataset = MAG240MMINIDataset(self.data_dir)
        else:
            dataset = MAG240MDataset(self.data_dir)
        return NeighborSampler(self.adj_t, node_idx=torch.from_numpy(np.arange(dataset.num_papers)),
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return NeighborSampler(self.adj_t, node_idx=self.val_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):  # Test best validation model once again.
        return NeighborSampler(self.adj_t, node_idx=self.val_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size, num_workers=4)

    def hidden_test_dataloader(self):
        return NeighborSampler(self.adj_t, node_idx=self.test_idx,
                               sizes=self.sizes, return_e_id=False,
                               transform=self.convert_batch,
                               batch_size=self.batch_size, num_workers=4)

    def convert_batch(self, batch_size, n_id, adjs):

        x = torch.from_numpy(self.x[n_id.numpy()]).to(torch.float)
        y = self.y[n_id[:batch_size]].to(torch.long)
        return Batch(x=x, y=y, adjs_t=[adj_t for adj_t, _, _ in adjs])


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
        # y_pred = y_pred.astype(np.short)

        if isinstance(y_pred_valid, torch.Tensor):
            y_pred_valid = y_pred_valid.cpu().numpy()
        # y_pred_valid = y_pred_valid.astype(np.short)

        makedirs(dir_path)
        filename = osp.join(dir_path, 'y_pred_mag240m')
        np.savez_compressed(filename, y_pred=y_pred, y_pred_valid=y_pred_valid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_channels', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--model', type=str, default='moco')
    parser.add_argument('--sizes', type=str, default='25-15')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--parallel', type=bool, default=False)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--resume', type=int, default=None)
    parser.add_argument('--valid_result', type=bool, default=False)
    parser.add_argument('--mini_graph', type=bool, default=False)
    parser.add_argument('--cs', type=bool, default=False)
    args = parser.parse_args()
    args.sizes = [int(i) for i in args.sizes.split('-')]
    print(args)

    seed_everything(42)
    datamodule = MAG240M(ROOT, args.batch_size, args.sizes, args.mini_graph)

    if not args.evaluate:
        model = Moco_v2(emb_dim=768)

        checkpoint_callback = ModelCheckpoint(monitor='val_acc', save_top_k=1)
        if args.parallel==True:
            gpus = [4,5,6,7]
            trainer = Trainer(gpus=gpus, max_epochs=args.epochs,
                              callbacks=[checkpoint_callback],
                              default_root_dir=f'logs/{args.model}')
        else:
            if args.resume==None:
                trainer = Trainer(gpus=args.device, max_epochs=args.epochs,
                                  callbacks=[checkpoint_callback],
                                  default_root_dir=f'logs/{args.model}')
            else:
                dirs = glob.glob(f'logs/{args.model}/lightning_logs/*')
                version = args.resume
                logdir = f'logs/{args.model}/lightning_logs/version_{version}'
                ckpt = glob.glob(f'{logdir}/checkpoints/*')[0]
                print('consume nodel version:',version)
                trainer = Trainer(gpus=args.device, resume_from_checkpoint=ckpt)

        trainer.fit(model, datamodule=datamodule)

    if args.evaluate:
        dirs = glob.glob(f'logs/{args.model}/lightning_logs/*')
        version = args.resume
        logdir = f'logs/{args.model}/lightning_logs/version_{version}'
        print(f'Evaluating saved model in {logdir}...')
        ckpt = glob.glob(f'{logdir}/checkpoints/*')[0]
        if args.parallel==True:
            gpus = [4,5,6,7]
            trainer = Trainer(gpus=gpus, resume_from_checkpoint=ckpt)
        else:
            trainer = Trainer(gpus=args.device, resume_from_checkpoint=ckpt)

        model = RGNN.load_from_checkpoint(
            checkpoint_path=ckpt, hparams_file=f'{logdir}/hparams.yaml').to(int(args.device))

        datamodule.batch_size = 16
        datamodule.sizes = [160] * len(args.sizes)  # (Almost) no sampling...

        # trainer.test(model=model, datamodule=datamodule)

        evaluator = MAG240MEvaluator()

        if args.valid_result:
            loader = datamodule.hidden_test_dataloader()
            loader1 = datamodule.val_dataloader()

            model.eval()
            y_preds = []
            y_preds_valid = []
            for batch in tqdm(loader):
                batch = batch.to(int(args.device))
                with torch.no_grad():
                    out = model(batch.x, batch.adjs_t).argmax(dim=-1).cpu()
                    y_preds.append(out)
            for batch in tqdm(loader1):
                batch = batch.to(int(args.device))
                with torch.no_grad():
                    out = model(batch.x, batch.adjs_t).argmax(dim=-1).cpu()
                    y_preds_valid.append(out)
            res = {'y_pred': torch.cat(y_preds, dim=0),'y_pred_valid': torch.cat(y_preds_valid, dim=0)}
            evaluator.save_test_submission(res, f'results/{args.model}')

        else:
            if args.cs:
                loader = datamodule.all_dataloader()

                model.eval()
                y_preds = []
                for batch in tqdm(loader):
                    batch = batch.to(int(args.device))
                    with torch.no_grad():
                        out = model(batch.x, batch.adjs_t).softmax(dim=-1).cpu()
                        # print(out)
                        y_preds.append(out)
                res = {'y_pred': torch.cat(y_preds, dim=0), 'y_pred_valid': torch.tensor([])}
                evaluator.save_test_submission(res, f'results/moco')

            else:
                loader = datamodule.hidden_test_dataloader()

                model.eval()
                y_preds = []
                for batch in tqdm(loader):
                    batch = batch.to(int(args.device))
                    with torch.no_grad():
                        out = model(batch.x, batch.adjs_t).argmax(dim=-1).cpu()
                        y_preds.append(out)
                res = {'y_pred': torch.cat(y_preds, dim=0), 'y_pred_valid': torch.tensor([])}
                evaluator.save_test_submission(res, f'results/{args.model}')
