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

ap_edge = np.load(f'{dataset.dir}/sorted_author_paper_edge.npy')
print('___________sub_train___________')
bias = 0
all_ = np.zeros(shape=(dataset.num_authors,dataset.num_classes))
for i in tqdm(range(train_idx.shape[0])):
    i = train_idx[i]
    for j in range(bias,ap_edge.shape[1]):
        if (i==ap_edge[1,j]):
            all_[int(ap_edge[0,j]),int(paper_label[ap_edge[1,j]])] += 1
        elif i<ap_edge[1,j]:
            bias = j
            break
all_2 = {}
for i in tqdm(range(all_.shape[0])):
    tmp = []
    for j in range(all_[i].shape[0]):
        if all_[i][j] != 0:
            tmp.append((j,all_[i][j]))
    all_2[i] = tmp

a_l = {}
bias = 0
for i in tqdm(range(train_idx.shape[0])):
    i = train_idx[i]
    for j in range(bias,ap_edge.shape[1]):
        if i==ap_edge[1,j]:
            if ap_edge[0,j] not in a_l.keys():
                a_l[ap_edge[0,j]] = [int(paper_label[ap_edge[1,j]])]
            else:
                a_l[ap_edge[0, j]].append(int(paper_label[ap_edge[1,j]]))
        elif i<ap_edge[1,j]:
            bias = j
            break
print(len(a_l.keys()))
ap_edge = dataset.edge_index('author', 'writes', 'paper')


#______________valid___________________
new_label = deepcopy(paper_label)
# ap_edge = dataset.edge_index('author', 'writes', 'paper')
def zero():
    return []
error_paper = {}
pa_dict = defaultdict(zero)
for i in tqdm(range(ap_edge.shape[1])):
    pa_dict[ap_edge[1, i]].append(all_2[ap_edge[0, i]])
relate = []
pred = []
paper_lis = list(coverage.keys())
for i in paper_lis:
    relate.append(i)
    counts = np.bincount(coverage[i])
    pred.append(np.argmax(counts))

true = new_label[relate]
true_error = {}
pred_error = {}
for i in range(len(pred)):
    if true[i] != pred[i]:
        error_paper[paper_lis[i]] = pa_dict[paper_lis[i]]
        true_error[paper_lis[i]] = true[i]
        pred_error[paper_lis[i]] = pred[i]
print('total:',c)
print(len(relate))
print('valid precision:',accuracy_score(true,pred))
for i,v in error_paper.items():
    print('_______________________________________________')
    print(i,v)
    print('true',true_error[i])
    print('pred', pred_error[i])







