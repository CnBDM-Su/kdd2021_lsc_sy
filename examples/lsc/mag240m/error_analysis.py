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
# a_l = softmax(a_l, axis=1)
# print(a_l)
# print((a_l.sum(1)!=0).sum())
# reliable_author = {}
# for i in a_l.keys():
#     if len(a_l[i]) > 1:
#         arr = np.array(a_l[i])
#
#         if arr[arr == a_l[i][0]].shape[0] >= np.round(arr.shape[0]*(4/5)):
#             counts = np.bincount(arr)
#             mode = np.argmax(counts)
#             reliable_author[i] = mode
#

# related_paper = []
# bias = 0
# keys = np.sort(list(reliable_author.keys()))
# for i in tqdm(range(len(reliable_author.keys()))):
#     i = keys[i]
#     for j in range(bias,ap_edge.shape[1]):
#         if i==ap_edge[0,j]:
#             related_paper.append(ap_edge[1, j])
#         elif i<ap_edge[0,j]:
#             bias = j
#             break
# print('related paper num:',len(related_paper))
# print('reliable author num:',len(reliable_author.keys()))
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
reliable_author = {}
for i in a_l.keys():
    if len(a_l[i]) > 1:
        arr = np.array(a_l[i])
        if arr[arr == a_l[i][0]].shape[0] == arr.shape[0]:
            counts = np.bincount(arr)
            mode = np.argmax(counts)
            reliable_author[i] = mode

print(len(reliable_author.keys()))
# print('___________sub_test___________')
# a_l_2 = {}
# bias = 0
# for i in tqdm(range(te_idx.shape[0])):
#     i = te_idx[i]
#     for j in range(bias,ap_edge.shape[1]):
#         if i==ap_edge[1,j]:
#             if ap_edge[0,j] not in a_l_2.keys():
#                 a_l_2[ap_edge[0,j]] = [paper_label[ap_edge[1,j]]]
#             else:
#                 a_l_2[ap_edge[0, j]].append(paper_label[ap_edge[1,j]])
#         elif i<ap_edge[1,j]:
#             bias = j
#             break
# print(len(a_l_2.keys()))
# reliable_author_2 = {}
# for i in a_l_2.keys():
#     if len(a_l_2[i]) > 1:
#         arr = np.array(a_l_2[i])
#         if arr[arr == a_l_2[i][0]].shape[0] == arr.shape[0]:
#             reliable_author_2[i] = a_l_2[i][0]
#
# print(len(reliable_author_2.keys()))

#______________valid___________________
new_label = deepcopy(paper_label)
# ap_edge = dataset.edge_index('author', 'writes', 'paper')
c = 0
coverage = {}
bias = 0
keys = np.sort(list(reliable_author.keys()))
for i in tqdm(range(len(reliable_author.keys()))):
    i = keys[i]
    l = reliable_author[i]
    for j in range(bias,ap_edge.shape[1]):
        if i==ap_edge[0,j]:
            c+=1
            if ap_edge[1, j] in valid_idx:
                if ap_edge[1, j] not in coverage.keys():
                    coverage[ap_edge[1, j]] = [l]
                else:
                    coverage[ap_edge[1, j]].append(l)
        elif i<ap_edge[0,j]:
            bias = j
            break
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

# # valid = deepcopy(paper_label)
# valid_related = []
# bias = 0
# c =0
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
# valid = np.zeros(shape=(valid_idx.shape[0],dataset.num_classes))
# for i in tqdm(range(valid_idx.shape[0])):
#     ind = valid_idx[i]
#     tmp = []
#     tmp_w = []
#     for j in range(bias, ap_edge.shape[1]):
#         if ind == ap_edge[1,j]:
#             tmp.append(a_l[ap_edge[0,j]])
#             # tmp_w.append(author_weight[ap_edge[0,j]])
#         elif ind < ap_edge[1,j]:
#             bias = j
#             break
#     if len(tmp)!=0:
#         c+=1
#         # tmp_w = np.array(softmax(tmp_w)).reshape(-1,1)
#         # valid[i] = np.mean(np.array(tmp)*tmp_w,0)
#         valid[i] = np.mean(np.array(tmp),0)
# print(c)
# np.save(f'{dataset.dir}/new_valid_label.npy',valid)





