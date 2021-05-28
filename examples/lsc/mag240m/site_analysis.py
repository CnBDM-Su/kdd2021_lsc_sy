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
    pp_edge = dataset.edge_index('paper', 'sites', 'paper')
    paper_label = dataset.paper_label
    year = dataset.all_paper_year
    def zero():
        return []
    pp_dict = defaultdict(zero)
    connect = []
    for i in tqdm(range(pp_edge.shape[1])):
        # pa_dict[ap_edge[1, i]].append(ap_edge[0, i])
        pp_dict[pp_edge[0, i]].append(pp_edge[1, i])
        pp_dict[pp_edge[1, i]].append(pp_edge[0, i])

    for i in pp_dict[args.author]:
        print(i,year[i],paper_label[i])