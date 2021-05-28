from tqdm import tqdm
import numpy as np
from copy import deepcopy
from root import ROOT
import torch
from ogb.utils.url import makedirs
from sklearn.metrics import accuracy_score,precision_score
from collections import defaultdict
import sys
sys.path.append('/var/ogb/ogb/lsc')
from mag240m_mini_graph import MAG240MMINIDataset
from scipy.special import softmax
from torch_sparse import SparseTensor
dataset = MAG240MMINIDataset(ROOT)

train_idx = dataset.get_idx_split('train')
# te_id = np.random.choice(train_idx.shape[0], size=(int(np.round(train_idx.shape[0]*0.2)),), replace=False)
# te_idx = np.sort(train_idx[te_id])
# train_idx = np.sort(np.array(list(set(train_idx) - set(te_idx))))
valid_idx = dataset.get_idx_split('valid')
test_idx = dataset.get_idx_split('test')
idx = np.concatenate([train_idx,valid_idx,test_idx],0)
paper_label = dataset.paper_label
year = dataset.all_paper_year
year_w = []
def zero():
    return []
ly_dict = defaultdict(zero)
for i in tqdm(range(dataset.num_papers)):
    ly_dict[paper_label[i]].append(year[i])

for i,v in ly_dict.items():
    print(i,min(v),max(v))