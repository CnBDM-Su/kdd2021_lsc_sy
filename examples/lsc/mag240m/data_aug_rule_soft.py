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

ap_edge = np.load(f'{dataset.dir}/sorted_author_paper_edge.npy')
print('___________sub_train___________')
bias = 0
a_l = np.zeros(shape=(dataset.num_authors,dataset.num_classes))
for i in tqdm(range(train_idx.shape[0])):
    i = train_idx[i]
    for j in range(bias,ap_edge.shape[1]):
        if i==ap_edge[1,j]:
            a_l[int(ap_edge[0,j]),int(paper_label[ap_edge[1,j]])] += 1
        elif i<ap_edge[1,j]:
            bias = j
            break
a_l = softmax(a_l, axis=1)
# print(a_l)
print((a_l.sum(1)!=0).sum())

valid_related = []
bias = 0
c =0
# def zero():
#     return []
# ap_dict = defaultdict(zero)
# connect = []
# for i in tqdm(range(ap_edge.shape[1])):
#     ap_dict[ap_edge[0, i]].append(ap_edge[1, i])
#
# author_weight = {}
# for i, v in tqdm(ap_dict.items()):
#     author_weight[i] = len(v)
#___________________soft_________________________

res = np.zeros(shape=(idx.shape[0],dataset.num_classes))
for i in tqdm(range(idx.shape[0])):
    ind = idx[i]
    tmp = []
    tmp_w = []
    for j in range(bias, ap_edge.shape[1]):
        if ind == ap_edge[1,j]:
            tmp.append(a_l[ap_edge[0,j]])
            # tmp_w.append(author_weight[ap_edge[0,j]])
        elif ind < ap_edge[1,j]:
            bias = j
            break
    if len(tmp)!=0:
        c+=1
        # tmp_w = np.array(softmax(tmp_w)).reshape(-1,1)
        # valid[i] = np.mean(np.array(tmp)*tmp_w,0)
        res[i] = np.mean(np.array(tmp),0)
print(c)

np.save(f'{dataset.dir}/new_all_label.npy',res)





