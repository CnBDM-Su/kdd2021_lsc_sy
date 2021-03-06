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
a_l = {}
for i in tqdm(range(train_idx.shape[0])):
    i = train_idx[i]
    for j in range(bias,ap_edge.shape[1]):
        if i==ap_edge[1,j]:
            if ap_edge[0, j] not in a_l.keys():
                a_l[ap_edge[0,j]] = [paper_label[ap_edge[1,j]]]
            else:
                a_l[ap_edge[0, j]].append(paper_label[ap_edge[1,j]])
        elif i<ap_edge[1,j]:
            bias = j
            break
# a_l = softmax(a_l, axis=1)
# print(a_l)
reliable_author = {}
for i in tqdm(a_l.keys()):
    if len(a_l[i]) > 1:
        arr = np.array(a_l[i]).astype(int)

        # if arr[arr == a_l[i][0]].shape[0] >= np.round(arr.shape[0]*(4/5)):
        counts = np.bincount(arr)
        mode = np.argmax(counts)
        if arr[arr == mode].shape[0] >= np.round(arr.shape[0]*(4/5)):
            reliable_author[i] = [mode,arr[arr == mode].shape[0]]

ap_edge = dataset.edge_index('author', 'writes', 'paper')
related_paper = []
bias = 0
keys = np.sort(list(reliable_author.keys()))
for i in tqdm(range(len(reliable_author.keys()))):
    i = keys[i]
    for j in range(bias,ap_edge.shape[1]):
        if i==ap_edge[0,j]:
            related_paper.append(ap_edge[1, j])
        elif i<ap_edge[0,j]:
            bias = j
            break
print('related paper num:',len(related_paper))
print('reliable author num:',len(reliable_author.keys()))


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
# print('___________valid___________')
# a_l_3 = {}
# bias = 0
# for i in tqdm(range(valid_idx.shape[0])):
#     i = valid_idx[i]
#     for j in range(bias,ap_edge.shape[1]):
#         if i==ap_edge[1,j]:
#             if ap_edge[0,j] not in a_l_3.keys():
#                 a_l_3[ap_edge[0,j]] = [paper_label[ap_edge[1,j]]]
#             else:
#                 a_l_3[ap_edge[0, j]].append(paper_label[ap_edge[1,j]])
#         elif i<ap_edge[1,j]:
#             bias = j
#             break
# print(len(a_l_3.keys()))
# reliable_author_3 = {}
# for i in a_l_3.keys():
#     if len(a_l_3[i]) > 1:
#         arr = np.array(a_l_3[i])
#         if arr[arr == a_l_3[i][0]].shape[0] == arr.shape[0]:
#             reliable_author_3[i] = a_l_3[i][0]
#
# print(len(reliable_author_3.keys()))
# print('___________test___________')
# a_l_4 = {}
# bias = 0
# for i in tqdm(range(test_idx.shape[0])):
#     i = test_idx[i]
#     for j in range(bias,ap_edge.shape[1]):
#         if i==ap_edge[1,j]:
#             if ap_edge[0,j] not in a_l_4.keys():
#                 a_l_4[ap_edge[0,j]] = [paper_label[ap_edge[1,j]]]
#             else:
#                 a_l_4[ap_edge[0, j]].append(paper_label[ap_edge[1,j]])
#         elif i<ap_edge[1,j]:
#             bias = j
#             break
# print(len(a_l_4.keys()))
# reliable_author_4 = {}
# for i in a_l_4.keys():
#     if len(a_l_4[i]) > 1:
#         arr = np.array(a_l_4[i])
#         if arr[arr==a_l_4[i][0]].shape[0]==arr.shape[0]:
#             reliable_author_4[i] = a_l_4[i][0]
#
# print(len(reliable_author_4.keys()))
#
# print('__________coverage__________')

# cover_1_1 = len(list(set(a_l.keys()) & set(a_l_2.keys())))/len(a_l_2.keys())
# cover_1_2 = len(list(set(a_l.keys()) & set(a_l_3.keys())))/len(a_l_3.keys())
# cover_1_3 = len(list(set(a_l.keys()) & set(a_l_4.keys())))/len(a_l_4.keys())
# print('all author sub_train & sub test coverage ratio:',cover_1_1)
# print('all author sub_train & valid coverage ratio:',cover_1_2)
# print('all author sub_train & test coverage ratio:',cover_1_3)
# cover_2_1 = len(list(set(reliable_author.keys()) & set(reliable_author_2.keys())))/len(reliable_author_2.keys())
# cover_2_2 = len(list(set(reliable_author.keys()) & set(reliable_author_3.keys())))/len(reliable_author_3.keys())
# cover_2_3 = len(list(set(reliable_author.keys()) & set(reliable_author_4.keys())))/len(reliable_author_4.keys())
# print('reliable author sub_train & sub test coverage ratio:',cover_2_1)
# print('reliable author sub_train & valid coverage ratio:',cover_2_2)
# print('reliable author sub_train & test coverage ratio:',cover_2_3)
# cover_3_1 = len(list(set(related_paper) & set(te_idx)))/te_idx.shape[0]
# cover_3_2 = len(list(set(related_paper) & set(valid_idx)))/valid_idx.shape[0]
# cover_3_3 = len(list(set(related_paper) & set(test_idx)))/test_idx.shape[0]
# print('related paper sub_train & sub test coverage ratio:',cover_3_1)
# print('related paper sub_train & valid coverage ratio:',cover_3_2)
# print('related paper sub_train & test coverage ratio:',cover_3_3)
# cover_4_1 = len(list(set(reliable_author.keys()) & set(a_l_2.keys())))/len(a_l_2.keys())
# cover_4_2 = len(list(set(reliable_author.keys()) & set(a_l_3.keys())))/len(a_l_3.keys())
# cover_4_3 = len(list(set(reliable_author.keys()) & set(a_l_4.keys())))/len(a_l_4.keys())
# print('reliable author sub_train & sub test coverage ratio:',cover_4_1)
# print('reliable author sub_train & valid coverage ratio:',cover_4_2)
# print('reliable author sub_train & test coverage ratio:',cover_4_3)
# ap_edge = dataset.edge_index('author', 'writes', 'paper')
#______________sub test___________________
# new_label = deepcopy(paper_label)
# c = 0
# coverage = {}
# bias = 0
# keys = np.sort(list(reliable_author.keys()))
# for i in tqdm(range(len(reliable_author.keys()))):
#     i = keys[i]
#     l = reliable_author[i]
#     for j in range(bias,ap_edge.shape[1]):
#         if i==ap_edge[0,j]:
#             c+=1
#             if ap_edge[1, j] in te_idx:
#                 if ap_edge[1, j] not in coverage.keys():
#                     coverage[ap_edge[1, j]] = [l]
#                 else:
#                     coverage[ap_edge[1, j]].append(l)
#         elif i<ap_edge[0,j]:
#             bias = j
#             break
# relate = []
# pred = []
# for i in coverage.keys():
#     relate.append(i)
#     counts = np.bincount(coverage[i])
#     pred.append(np.argmax(counts))
#
# true = new_label[relate]
# print('total:',c)
# print(len(relate))
# print('sub_test precision:',accuracy_score(true,pred))

#______________valid___________________
new_label = deepcopy(paper_label)
c = 0
def zero():
    return []
coverage = defaultdict(zero)
bias = 0
keys = np.sort(list(reliable_author.keys()))
for i in tqdm(range(len(reliable_author.keys()))):
    i = keys[i]
    # l = reliable_author[i][0]
    # num = reliable_author[i][1]
    for j in range(bias,ap_edge.shape[1]):
        if i==ap_edge[0,j]:
            c+=1
            if ap_edge[1, j] in valid_idx:
                coverage[ap_edge[1, j]].append(reliable_author[i])
        elif i<ap_edge[0,j]:
            bias = j
            break

relate = []
pred = []
for i in coverage.keys():
    relate.append(i)
    count = {}
    for j in coverage[i]:
        if j[0] not in count.keys():
            count[j[0]] = j[1]
        else:
            count[j[0]] += j[1]
    new_label[i] = max(count.items(), key=lambda x: x[1])[0]
    # pred.append(max(count.items(), key=lambda x: x[1])[0])
    # counts = np.bincount(coverage[i])
    # pred.append(np.argmax(counts))

# true = new_label[relate]
np.save(f'{dataset.dir}/data_rule_result_relate.npy',np.array(relate))
np.save(f'{dataset.dir}/data_rule_result.npy',new_label)
print('total:',c)
print(len(relate))
# print('valid precision:',accuracy_score(true,pred))

#______________predict________________
# new_label = deepcopy(paper_label)
# new_tr = []
# bias = 0
# keys = np.sort(list(reliable_author.keys()))
# for i in tqdm(range(len(reliable_author.keys()))):
#     i = keys[i]
#     l = reliable_author[i]
#     for j in range(bias,ap_edge.shape[1]):
#         if i==ap_edge[0,j]:
#             if ap_edge[1,j] not in idx:
#                 new_tr.append(ap_edge[1,j])
#                 new_label[ap_edge[1,j]] = l
#         elif i<ap_edge[0,j]:
#             bias = j
#             break
# print('new label num:',len(new_tr))
# new_tr = np.sort(train_idx.tolist() + new_tr)
# np.save(f'{dataset.dir}/new_train_idx.npy',new_tr)
# np.save(f'{dataset.dir}/new_paper_label.npy',new_label)

# ______________predict_valid____________
# valid = deepcopy(paper_label)
# valid_related = []
# bias = 0
# coverage = {}
# keys = np.sort(list(reliable_author.keys()))
# for i in tqdm(range(len(reliable_author.keys()))):
#     i = keys[i]
#     l = reliable_author[i]
#     for j in range(bias,ap_edge.shape[1]):
#         if i==ap_edge[0,j]:
#             if ap_edge[1,j] in valid_idx:
#                 if ap_edge[1, j] not in coverage.keys():
#                     coverage[ap_edge[1, j]] = [l]
#                 else:
#                     coverage[ap_edge[1, j]].append(l)
#         elif i<ap_edge[0,j]:
#             bias = j
#             break
# # print('new label num:',len(new_tr))
# valid_related = []
# for i in coverage.keys():
#     valid_related.append(i)
#     counts = np.bincount(coverage[i])
#     valid[i] = np.argmax(counts)
#
# valid_related = np.array(valid_related)
# print(valid_related.shape)
# np.save(f'{dataset.dir}/changed_valid_idx.npy',valid_related)
# np.save(f'{dataset.dir}/new_valid_label.npy',valid)

# valid = deepcopy(paper_label)
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






