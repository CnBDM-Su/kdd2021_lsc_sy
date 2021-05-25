from tqdm import tqdm
import numpy as np
from copy import deepcopy
from root import ROOT
from ogb.utils.url import makedirs
from sklearn.metrics import accuracy_score,precision_score
import sys
sys.path.append('/var/ogb/ogb/lsc')
from mag240m_mini_graph import MAG240MMINIDataset


dataset = MAG240MMINIDataset(ROOT)

train_idx = dataset.get_idx_split('train')
te_id = np.random.choice(np.range(train_idx.shape[0]), size=(np.round(train_idx.shape[0]*0.2),), replace=False)
te_idx = train_idx[te_id]
train_idx = np.array(list(set(train_idx) - set(te_idx)))
valid_idx = dataset.get_idx_split('valid')
test_idx = dataset.get_idx_split('test')
idx = np.concatenate([train_idx,valid_idx,test_idx],0)
paper_label = dataset.paper_label

ap_edge = np.load(f'{dataset.dir}/sorted_author_paper_edge.npy')
a_l = {}
bias = 0
for i in tqdm(range(train_idx.shape[0])):
    i = train_idx[i]
    for j in range(bias,ap_edge.shape[1]):
        if i==ap_edge[1,j]:
            if ap_edge[0,j] not in a_l.keys():
                a_l[ap_edge[0,j]] = [paper_label[ap_edge[1,j]]]
            else:
                a_l[ap_edge[0, j]].append(paper_label[ap_edge[1,j]])
        elif i<ap_edge[1,j]:
            bias = j
            break
print(len(a_l.keys()))
reliable_author = {}
for i in a_l.keys():
    if len(a_l[i]) > 1:
        if np.mean(a_l[i])==a_l[i][0]:
            reliable_author[i] = a_l[i][0]

print(len(reliable_author.keys()))
ap_edge = dataset.edge_index('author', 'writes', 'paper')
#______________test___________________
new_label = deepcopy(paper_label)
true = new_label[te_idx]
bias = 0
for i in tqdm(len(reliable_author.keys())):
    i = reliable_author.keys()[i]
    l = reliable_author[i]
    for j in range(bias,ap_edge.shape[1]):
        if i==ap_edge[0,j]:
                new_label[ap_edge[1,j]] = l
        elif i<ap_edge[0,j]:
            bias = j
            break
pred = new_label[te_idx]
print(precision_score(true,pred))

#______________predict________________
# new_label = deepcopy(paper_label)
# bias = 0
# for i in tqdm(len(reliable_author.keys())):
#     i = reliable_author.keys()[i]
#     l = reliable_author[i]
#     for j in range(bias,ap_edge.shape[1]):
#         if i==ap_edge[0,j]:
#             if ap_edge[1,j] not in idx:
#                 new_label[ap_edge[1,j]] = l
#         elif i<ap_edge[0,j]:
#             bias = j
#             break
# np.save(f'{dataset.dir}/new_paper_label.npy',new_label)
