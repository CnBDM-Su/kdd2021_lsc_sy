import time
import argparse
from tqdm import tqdm
import torch
import os.path as osp
# from ogb.lsc import MAG240MDataset
from root import ROOT
import numpy as np
import sys
sys.path.append('/var/ogb/ogb/lsc')
from mag240m_mini_graph import MAG240MMINIDataset
from ogb.utils.url import makedirs
from copy import deepcopy
import glob
from final_mlp import
from final_rgat

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
    parser.add_argument('--evaluate', type=int, default=0)

    #mlp_parameter
    parser.add_argument('--hidden_channels', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=2),
    parser.add_argument('--no_batch_norm', action='store_true')
    parser.add_argument('--relu_last', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=380000)
    parser.add_argument('--epochs', type=int, default=1000)

    # rgat_parameter
    parser.add_argument('--hidden_channels', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--model', type=str, default='rgat',
                        choices=['rgat', 'rgraphsage'])
    parser.add_argument('--sizes', type=str, default='25-15')
    parser.add_argument('--resume', type=int, default=None)

    # cs_parameter
    parser.add_argument('--num_correction_layers', type=int, default=3)
    parser.add_argument('--correction_alpha', type=float, default=1.0)
    parser.add_argument('--num_smoothing_layers', type=int, default=2)
    parser.add_argument('--smoothing_alpha', type=float, default=0.8)


    args = parser.parse_args()
    args.sizes = [int(i) for i in args.sizes.split('-')]
    print(args)

    torch.manual_seed(12345)
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

    if args.evaluate == 0:
        np.random.seed(0)
        valid_idx_ = np.random.choice(valid_idx, size=(int(valid_idx.shape[0]*ratio),), replace=False)
        np.save(f'{dataset.dir}/val_idx_'+str(ratio)+'.npy',valid_idx_)
    else:
        valid_idx_ = np.load(f'{dataset.dir}/val_idx_'+str(ratio)+'.npy')
    train_idx = np.concatenate([train_idx,valid_idx_],0)
    valid_idx = np.array(list(set(valid_idx) - set(valid_idx_)))

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
    if args.evaluate ==0:
        # dataset.num_paper_features
        model = MLP(dataset.num_paper_features*2, args.hidden_channels,
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
        model = MLP(dataset.num_paper_features*2, args.hidden_channels,
                    dataset.num_classes, args.num_layers, args.dropout,
                    not args.no_batch_norm, args.relu_last).to(device)
        if args.parallel == True:
            model = torch.nn.DataParallel(model, device_ids=gpus)
        model.load_state_dict(torch.load('results/mlp/model.pkl'))
#___________________predict______________________________
        feat = x
        w = torch.t(model.state_dict()['module.lins.0.weight']).to(device).to(torch.half)
        bias = model.state_dict()['module.lins.0.bias'].to(device).to(torch.half)
        print(w.shape)
        print(bias.shape)
        batch_size = 1000
        con = []
        for i in tqdm(range(feat.shape[0]//batch_size+1)):
            end = min((i+1)*batch_size,feat.shape[0])
            feat1 = torch.from_numpy(feat[i*batch_size:end]).to(device).to(torch.half)
            res = (torch.matmul(feat1,w)+bias).cpu()
            con.append(res)

        con = torch.cat(con).cpu().numpy()
        from sklearn.preprocessing import MinMaxScaler
        mm = MinMaxScaler((-1,1))
        con =mm.fit_transform(con)
        print(con.shape)
        print(con)
        np.save(f'{dataset.dir}/256dim_rgat_twice_256/node_feat.npy',con)


        seed_everything(42)
        datamodule = MAG240M(ROOT, args.batch_size, args.sizes, args.mini_graph)

        if not args.evaluate:
            model = RGNN(args.model, datamodule.num_features,
                         datamodule.num_classes, args.hidden_channels,
                         datamodule.num_relations, num_layers=len(args.sizes),
                         dropout=args.dropout)
            print(f'#Params {sum([p.numel() for p in model.parameters()])}')
            checkpoint_callback = ModelCheckpoint(monitor='val_acc', save_top_k=1)
            if args.parallel == True:
                gpus = [4, 5, 6, 7]
                trainer = Trainer(gpus=gpus, max_epochs=args.epochs,
                                  callbacks=[checkpoint_callback],
                                  default_root_dir=f'logs/{args.model}')
            else:
                if args.resume == None:
                    trainer = Trainer(gpus=args.device, max_epochs=args.epochs,
                                      callbacks=[checkpoint_callback],
                                      default_root_dir=f'logs/{args.model}')
                else:
                    dirs = glob.glob(f'logs/{args.model}/lightning_logs/*')
                    version = args.resume
                    logdir = f'logs/{args.model}/lightning_logs/version_{version}'
                    ckpt = glob.glob(f'{logdir}/checkpoints/*')[0]
                    print('consume nodel version:', version)
                    trainer = Trainer(gpus=args.device, resume_from_checkpoint=ckpt)

            trainer.fit(model, datamodule=datamodule)

        if args.evaluate:
            dirs = glob.glob(f'logs/{args.model}/lightning_logs/*')
            version = args.resume
            logdir = f'logs/{args.model}/lightning_logs/version_{version}'
            print(f'Evaluating saved model in {logdir}...')
            ckpt = glob.glob(f'{logdir}/checkpoints/*')[0]
            if args.parallel == True:
                gpus = [4, 5, 6, 7]
                trainer = Trainer(gpus=gpus, resume_from_checkpoint=ckpt)
            else:
                trainer = Trainer(gpus=args.device, resume_from_checkpoint=ckpt)

            model = RGNN.load_from_checkpoint(
                checkpoint_path=ckpt, hparams_file=f'{logdir}/hparams.yaml').to(int(args.device))

            datamodule.batch_size = 16
            datamodule.sizes = [160] * len(args.sizes)  # (Almost) no sampling...

            # trainer.test(model=model, datamodule=datamodule)

            evaluator = MAG240MEvaluator()

            loader = datamodule.all_dataloader()
            # loader = datamodule.val_dataloader()

            model.eval()
            y_preds = []
            for batch in tqdm(loader):
                batch = batch.to(int(args.device))
                with torch.no_grad():
                    out = model(batch.x, batch.adjs_t).softmax(dim=-1).cpu()
                    # print(out)
                    y_preds.append(out)
            res = {'y_pred': torch.cat(y_preds, dim=0)}
            evaluator.save_test_submission(res, f'results/rgat_cs_v95')


        evaluator = MAG240MEvaluator()

