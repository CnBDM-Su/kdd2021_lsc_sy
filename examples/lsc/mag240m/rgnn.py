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


def get_col_slice(x, start_row_idx, end_row_idx, start_col_idx, end_col_idx):
    outs = []
    chunk = 100000
    for i in tqdm(range(start_row_idx, end_row_idx, chunk)):
        j = min(i + chunk, end_row_idx)
        outs.append(x[i:j, start_col_idx:end_col_idx].copy())
    return np.concatenate(outs, axis=0)


def save_col_slice(x_src, x_dst, start_row_idx, end_row_idx, start_col_idx,
                   end_col_idx):
    assert x_src.shape[0] == end_row_idx - start_row_idx
    assert x_src.shape[1] == end_col_idx - start_col_idx
    chunk, offset = 100000, start_row_idx
    for i in tqdm(range(0, end_row_idx - start_row_idx, chunk)):
        j = min(i + chunk, end_row_idx - start_row_idx)
        x_dst[offset + i:offset + j, start_col_idx:end_col_idx] = x_src[i:j]


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

    def prepare_data(self):
        if self.mini:
            dataset = MAG240MMINIDataset(self.data_dir)
        else:
            dataset = MAG240MDataset(self.data_dir)

        path = f'{dataset.dir}/paper_to_paper_symmetric.pt'
        if not osp.exists(path):  # Will take approximately 5 minutes...
            t = time.perf_counter()
            print('Converting adjacency matrix...', end=' ', flush=True)
            edge_index = dataset.edge_index('paper', 'cites', 'paper')
            edge_index = torch.from_numpy(edge_index)
            adj_t = SparseTensor(
                row=edge_index[0], col=edge_index[1],
                sparse_sizes=(dataset.num_papers, dataset.num_papers),
                is_sorted=True)
            torch.save(adj_t.to_symmetric(), path)
            print(f'Done! [{time.perf_counter() - t:.2f}s]')

        path = f'{dataset.dir}/full_adj_t.pt'
        if not osp.exists(path):  # Will take approximately 16 minutes...
            t = time.perf_counter()
            print('Merging adjacency matrices...', end=' ', flush=True)

            row, col, _ = torch.load(
                f'{dataset.dir}/paper_to_paper_symmetric.pt').coo()
            rows, cols = [row], [col]

            edge_index = dataset.edge_index('author', 'writes', 'paper')
            row, col = torch.from_numpy(edge_index)
            row += dataset.num_papers
            rows += [row, col]
            cols += [col, row]

            edge_index = dataset.edge_index('author', 'institution')
            row, col = torch.from_numpy(edge_index)
            row += dataset.num_papers
            col += dataset.num_papers + dataset.num_authors
            rows += [row, col]
            cols += [col, row]

            edge_types = [
                torch.full(x.size(), i, dtype=torch.int8)
                for i, x in enumerate(rows)
            ]

            row = torch.cat(rows, dim=0)
            del rows
            col = torch.cat(cols, dim=0)
            del cols

            N = (dataset.num_papers + dataset.num_authors +
                 dataset.num_institutions)

            perm = (N * row).add_(col).numpy().argsort()
            perm = torch.from_numpy(perm)
            row = row[perm]
            col = col[perm]

            edge_type = torch.cat(edge_types, dim=0)[perm]
            del edge_types

            full_adj_t = SparseTensor(row=row, col=col, value=edge_type,
                                      sparse_sizes=(N, N), is_sorted=True)

            torch.save(full_adj_t, path)
            print(f'Done! [{time.perf_counter() - t:.2f}s]')

        path = f'{dataset.dir}/full_feat.npy'
        done_flag_path = f'{dataset.dir}/full_feat_done.txt'
        if not osp.exists(done_flag_path):  # Will take ~3 hours...
            t = time.perf_counter()
            print('Generating full feature matrix...')

            node_chunk_size = 100000
            dim_chunk_size = 64
            N = (dataset.num_papers + dataset.num_authors +
                 dataset.num_institutions)

            paper_feat = dataset.paper_feat
            x = np.memmap(path, dtype=np.float16, mode='w+',
                          shape=(N, self.num_features))

            print('Copying paper features...')
            for i in tqdm(range(0, dataset.num_papers, node_chunk_size)):
                j = min(i + node_chunk_size, dataset.num_papers)
                x[i:j] = paper_feat[i:j]

            edge_index = dataset.edge_index('author', 'writes', 'paper')
            row, col = torch.from_numpy(edge_index)
            adj_t = SparseTensor(
                row=row, col=col,
                sparse_sizes=(dataset.num_authors, dataset.num_papers),
                is_sorted=True)

            # Processing 64-dim subfeatures at a time for memory efficiency.
            print('Generating author features...')
            for i in tqdm(range(0, self.num_features, dim_chunk_size)):
                j = min(i + dim_chunk_size, self.num_features)
                inputs = get_col_slice(paper_feat, start_row_idx=0,
                                       end_row_idx=dataset.num_papers,
                                       start_col_idx=i, end_col_idx=j)
                inputs = torch.from_numpy(inputs)
                outputs = adj_t.matmul(inputs, reduce='mean').numpy()
                del inputs
                save_col_slice(
                    x_src=outputs, x_dst=x, start_row_idx=dataset.num_papers,
                    end_row_idx=dataset.num_papers + dataset.num_authors,
                    start_col_idx=i, end_col_idx=j)
                del outputs

            edge_index = dataset.edge_index('author', 'institution')
            row, col = torch.from_numpy(edge_index)
            adj_t = SparseTensor(
                row=col, col=row,
                sparse_sizes=(dataset.num_institutions, dataset.num_authors),
                is_sorted=False)

            print('Generating institution features...')
            # Processing 64-dim subfeatures at a time for memory efficiency.
            for i in tqdm(range(0, self.num_features, dim_chunk_size)):
                j = min(i + dim_chunk_size, self.num_features)
                inputs = get_col_slice(
                    x, start_row_idx=dataset.num_papers,
                    end_row_idx=dataset.num_papers + dataset.num_authors,
                    start_col_idx=i, end_col_idx=j)
                inputs = torch.from_numpy(inputs)
                outputs = adj_t.matmul(inputs, reduce='mean').numpy()
                del inputs
                save_col_slice(
                    x_src=outputs, x_dst=x,
                    start_row_idx=dataset.num_papers + dataset.num_authors,
                    end_row_idx=N, start_col_idx=i, end_col_idx=j)
                del outputs

            x.flush()
            del x
            print(f'Done! [{time.perf_counter() - t:.2f}s]')

            with open(done_flag_path, 'w') as f:
                f.write('done')

    def setup(self, stage: Optional[str] = None):
        t = time.perf_counter()
        print('Reading dataset...', end=' ', flush=True)
        if self.mini:
            dataset = MAG240MMINIDataset(self.data_dir)
        else:
            dataset = MAG240MDataset(self.data_dir)

        np.random.seed(0)
        train_idx = dataset.get_idx_split('train')
        valid_idx = dataset.get_idx_split('valid')
        test_idx = dataset.get_idx_split('test')
        # valid_idx_ = np.random.choice(valid_idx, size=(int(valid_idx.shape[0] * ratio),), replace=False)
        # valid_idx_ = np.load(f'{dataset.dir}/val_idx_1.0.npy')
        # np.save(f'{dataset.dir}/val_idx_' + str(ratio) + '.npy', valid_idx_)
        # valid_idx_ = np.load(f'{dataset.dir}/val_idx_' + str(ratio) + '.npy')

        # train_idx = np.concatenate([train_idx, valid_idx_], 0)
        # valid_idx = np.array(list(set(valid_idx) - set(valid_idx_)))

        self.train_idx = torch.from_numpy(train_idx)
        self.train_idx = self.train_idx
        self.train_idx.share_memory_()
        self.val_idx = torch.from_numpy(valid_idx)
        self.val_idx.share_memory_()
        self.test_idx = torch.from_numpy(test_idx)
        self.test_idx.share_memory_()
        self.idx = torch.cat([self.train_idx,self.val_idx,self.test_idx],0)

        N = dataset.num_papers + dataset.num_authors + dataset.num_institutions

        # self.x = {}
        # for i in range(1000):
        #     self.x[i] = np.memmap(f'{dataset.dir}/full_feat_split/full_feat_'+str(i)+'.npy', dtype=np.float16,
        #                    mode='r', shape=(N//1000, self.num_features))
        # self.x[1000] = np.memmap(f'{dataset.dir}/full_feat_split/full_feat_' + str(i) + '.npy', dtype=np.float16,
        #                       mode='r', shape=(N-1000*(N//1000), self.num_features))
        # self.x = zarr.open(f'{dataset.dir}/full_feat.zarr', mode='r',shape=(N, self.num_features) ,
        #                    chunks=(200000, self.num_features), dtype=np.float16)
        self.x = np.memmap('/var/kdd-data/mag240m_kddcup2021/mini_graph/full_feat.npy', dtype=np.float16,
                           mode='r', shape=(N, 256))
        # self.x = np.load('/var/kdd-data/mag240m_kddcup2021/mini_graph/1024dim_256/full_feat.npy')
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
                               batch_size=self.batch_size, num_workers=16)
    def related_dataloader(self):
        return NeighborSampler(self.adj_t, node_idx=self.idx,
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
        # t = time.perf_counter()
        # x = []
        #
        # for i in n_id.numpy():
        #     x.append(self.x[i//self.file_batch_size][i%self.file_batch_size])
        # x = torch.from_numpy(np.array(x)).to(torch.float)
        # print(n_id.shape)
        # x = torch.from_numpy(self.x.get_orthogonal_selection((n_id.numpy(), slice(None)))).to(
        #     torch.float)
        x = torch.from_numpy(self.x[n_id.numpy()]).to(torch.float)
        # print(sys.getsizeof(x.storage()))
        y = self.y[n_id[:batch_size]].to(torch.long)
        # print(f'Done sampling! [{time.perf_counter() - t:.2f}s]')
        return Batch(x=x, y=y, adjs_t=[adj_t for adj_t, _, _ in adjs])


class RGNN(LightningModule):
    def __init__(self, model: str, in_channels: int, out_channels: int,
                 hidden_channels: int, num_relations: int, num_layers: int,
                 heads: int = 4, dropout: float = 0.5):
        super().__init__()
        self.save_hyperparameters()
        self.model = model.lower()
        self.num_relations = num_relations
        self.dropout = dropout

        self.convs = ModuleList()
        self.norms = ModuleList()
        self.skips = ModuleList()

        if self.model == 'rgat':
            self.convs.append(
                ModuleList([
                    GATConv(in_channels, hidden_channels // heads, heads,
                            add_self_loops=False) for _ in range(num_relations)
                ]))

            for _ in range(num_layers - 1):
                self.convs.append(
                    ModuleList([
                        GATConv(hidden_channels, hidden_channels // heads,
                                heads, add_self_loops=False)
                        for _ in range(num_relations)
                    ]))

        elif self.model == 'rgraphsage':
            self.convs.append(
                ModuleList([
                    SAGEConv(in_channels, hidden_channels, root_weight=False)
                    for _ in range(num_relations)
                ]))

            for _ in range(num_layers - 1):
                self.convs.append(
                    ModuleList([
                        SAGEConv(hidden_channels, hidden_channels,
                                 root_weight=False)
                        for _ in range(num_relations)
                    ]))

        for _ in range(num_layers):
            self.norms.append(BatchNorm1d(hidden_channels))

        self.skips.append(Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.skips.append(Linear(hidden_channels, hidden_channels))

        self.mlp = Sequential(
            Linear(hidden_channels, hidden_channels),
            BatchNorm1d(hidden_channels),
            ReLU(inplace=True),
            Dropout(p=self.dropout),
            Linear(hidden_channels, out_channels)
            # Linear(hidden_channels, 256),
            # Linear(256, out_channels),
        )

        self.acc = Accuracy()

    def forward(self, x: Tensor, adjs_t: List[SparseTensor]) -> Tensor:
        for i, adj_t in enumerate(adjs_t):
            x_target = x[:adj_t.size(0)]

            out = self.skips[i](x_target)
            for j in range(self.num_relations):
                edge_type = adj_t.storage.value() == j
                subadj_t = adj_t.masked_select_nnz(edge_type, layout='coo')
                if subadj_t.nnz() > 0:
                    out += self.convs[i][j]((x, x_target), subadj_t)

            x = self.norms[i](out)
            x = F.elu(x) if self.model == 'rgat' else F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.mlp(x)

    def infer(self, x: Tensor, adjs_t: List[SparseTensor]) -> Tensor:
        for i, adj_t in enumerate(adjs_t):
            x_target = x[:adj_t.size(0)]

            out = self.skips[i](x_target)
            for j in range(self.num_relations):
                edge_type = adj_t.storage.value() == j
                subadj_t = adj_t.masked_select_nnz(edge_type, layout='coo')
                if subadj_t.nnz() > 0:
                    out += self.convs[i][j]((x, x_target), subadj_t)

            x = self.norms[i](out)
            x = F.elu(x) if self.model == 'rgat' else F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.mlp[:-1](x)

    def training_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.adjs_t)
        train_loss = F.cross_entropy(y_hat, batch.y)
        train_acc = self.acc(y_hat.softmax(dim=-1), batch.y)
        self.log('train_acc', train_acc, prog_bar=True, on_step=False,
                 on_epoch=True)
        return train_loss

    def validation_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.adjs_t)
        val_acc = self.acc(y_hat.softmax(dim=-1), batch.y)
        self.log('val_acc', val_acc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return val_acc

    def test_step(self, batch, batch_idx: int):
        y_hat = self(batch.x, batch.adjs_t)
        test_acc = self.acc(y_hat.softmax(dim=-1), batch.y)
        self.log('test_acc', test_acc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return test_acc

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=25, gamma=0.25)
        return [optimizer], [scheduler]

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
    parser.add_argument('--model', type=str, default='rgat',
                        choices=['rgat', 'rgraphsage'])
    parser.add_argument('--sizes', type=str, default='25-15')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--parallel', type=bool, default=False)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--resume', type=int, default=None)
    parser.add_argument('--valid_result', type=bool, default=False)
    parser.add_argument('--mini_graph', type=bool, default=False)
    parser.add_argument('--cs', type=bool, default=False)
    parser.add_argument('--cut_hidden', type=bool, default=False)
    args = parser.parse_args()
    args.sizes = [int(i) for i in args.sizes.split('-')]
    print(args)

    seed_everything(42)
    datamodule = MAG240M(ROOT, args.batch_size, args.sizes, args.mini_graph)

    if not args.evaluate:
        model = RGNN(args.model, datamodule.num_features,
                     datamodule.num_classes, args.hidden_channels,
                     datamodule.num_relations, num_layers=len(args.sizes),
                     dropout=args.dropout)
        print(f'#Params {sum([p.numel() for p in model.parameters()])}')
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
        if args.cut_hidden == True:
            loader1 = datamodule.all_dataloader()
            model.eval()
            y_preds = []
            for batch in tqdm(loader1):
                batch = batch.to(int(args.device))
                with torch.no_grad():
                    out = model.infer(batch.x, batch.adjs_t).cpu().numpy()
                    y_preds.append(out)
            # print
            y_preds = np.concatenate(y_preds)
            np.save('/var/kdd-data/mag240m_kddcup2021/mini_graph/256dim_rgat_val/node_feat.npy',y_preds)

        elif args.valid_result:
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
                # loader = datamodule.all_dataloader()
                loader = datamodule.val_dataloader()

                model.eval()
                y_preds = []
                for batch in tqdm(loader):
                    batch = batch.to(int(args.device))
                    with torch.no_grad():
                        out = model(batch.x, batch.adjs_t).softmax(dim=-1).cpu()
                        # print(out)
                        y_preds.append(out)
                res = {'y_pred': torch.cat(y_preds, dim=0), 'y_pred_valid': torch.tensor([])}
                evaluator.save_test_submission(res, f'results/rgat_cs_v3')

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
