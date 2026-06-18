import os
import numpy as np
from math import sqrt
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import random

class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='',
                 xd=None, xt=None, y=None, z=None, transform=None,
                 pre_transform=None, smile_graph=None):

        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt, y, z, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, xd, xt, y, z, smile_graph):
        count = 0
        print(len(xd), len(xt), '====', len(y))
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        for i in range(data_len):
            print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smiles = xd[i]
            target = xt[i]
            labels = y[i]
            seqdrug = z[i]
            
            c_size, features, edge_index = smile_graph[smiles]
            
            if len(edge_index) == 0:
                count += 1
                print(f'No edges for graph {i + 1}, skipping...', smiles)
                continue

            GCNData = DATA.Data(
                x=torch.Tensor(features),
                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                y=torch.LongTensor([labels])
            )
            GCNData.target = torch.LongTensor([target])
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            GCNData.__setitem__('seqdrug', torch.FloatTensor([seqdrug]))
            data_list.append(GCNData)
        
        print("Removed irregular graphs:", count, "Total graphs:", data_len)
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
            
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])