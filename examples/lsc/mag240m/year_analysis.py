from tqdm import tqdm
import numpy as np
from copy import deepcopy
from root import ROOT
import argparse
from collections import defaultdict
import sys
sys.path.append('/var/ogb/ogb/lsc')
from mag240m_mini_graph import MAG240MMINIDataset
# te_id = np.random.choice(train_idx.shape[0], size=(int(np.round(train_idx.shape[0]*0.2)),), replace=False)
# te_idx = np.sort(train_idx[te_id])
# train_idx = np.sort(np.array(list(set(train_idx) - set(te_idx))))

# year_w = []
# def zero():
#     return []
# ly_dict = defaultdict(zero)
# for i in tqdm(train_idx):
#     ly_dict[paper_label[i]].append(year[i])
#
# for i,v in ly_dict.items():
#     print(i,min(v),max(v))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--author', type=int, default=0)

    args = parser.parse_args()
    dataset = MAG240MMINIDataset(ROOT)
    ap_edge = dataset.edge_index('author', 'writes', 'paper')
    paper_label = dataset.paper_label
    year = dataset.all_paper_year
    def zero():
        return []
    pa_dict = defaultdict(zero)
    ap_dict = defaultdict(zero)
    connect = []
    for i in tqdm(range(ap_edge.shape[1])):
        # pa_dict[ap_edge[1, i]].append(ap_edge[0, i])
        ap_dict[ap_edge[0, i]].append(ap_edge[1, i])

    for i in ap_dict[args.author]:
        print(i,year[i])