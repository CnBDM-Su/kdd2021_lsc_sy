from tqdm import tqdm
import numpy as np
from copy import deepcopy
from root import ROOT
import argparse
from collections import defaultdict
import sys
sys.path.append('/var/ogb/ogb/lsc')
from mag240m_mini_graph import MAG240MMINIDataset

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
        print(i,year[i],paper_label[i])