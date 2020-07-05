# -*- coding: utf-8 -*-
"""
Created on 17/9/2019
@author: RuihongQiu
"""

import torch
import collections
import logging
from torch_geometric.data import InMemoryDataset, Data
from se_data_process import load_data_valid, load_testdata


class MultiSessionsGraph(InMemoryDataset):
    """Every session is a graph."""
    def __init__(self, root, phrase=None, transform=None, pre_transform=None, sampled_data=None):
        """
        Args:
            root: address of the dataset
            phrase: 'train' or 'test1' ~ 'test5' or 'sampled***' or 'uni***'
        """
        self.phrase, self.sampled_data = phrase, sampled_data
        logging.warning(self.phrase)
        super(MultiSessionsGraph, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
     
    @property
    def raw_file_names(self):
        return [self.phrase + '.txt.csv']
    
    @property
    def processed_file_names(self):
        return [self.phrase + '.pt']
    
    def download(self):
        pass
    
    def process(self):
        # data = [[x], [y], [user]]
        if self.sampled_data is not None:
            data = self.sampled_data
        else:
            if self.phrase == 'train':
                data, valid = load_data_valid(self.raw_dir + '/' + self.raw_file_names[0], 0)
            else:
                data = load_testdata(self.raw_dir + '/' + self.raw_file_names[0])
        data_list = []
        for sequence, y, userid in zip(data[0], data[1], data[2]):
            count = collections.Counter(sequence)
            i = 0
            nodes = {}    # dict{15: 0, 16: 1, 18: 2, ...}
            senders = []
            x = []
            for node in sequence:
                if node not in nodes:
                    nodes[node] = i
                    x.append([node])
                    i += 1
                senders.append(nodes[node])
            receivers = senders[:]
            num_count = [count[i[0]] for i in x]

            if len(senders) != 1:
                del senders[-1]  # the last item is a receiver
                del receivers[0]  # the first item is a sender

            pair = {}
            sur_senders = senders[:]
            sur_receivers = receivers[:]
            i = 0
            for sender, receiver in zip(sur_senders, sur_receivers):
                if str(sender) + '-' + str(receiver) in pair:
                    pair[str(sender) + '-' + str(receiver)] += 1
                    del senders[i]
                    del receivers[i]
                else:
                    pair[str(sender) + '-' + str(receiver)] = 1
                    i += 1

            count = collections.Counter(senders)
            out_degree_inv = [1 / count[i] for i in senders]

            count = collections.Counter(receivers)
            in_degree_inv = [1 / count[i] for i in receivers]
            
            in_degree_inv = torch.tensor(in_degree_inv, dtype=torch.float)
            out_degree_inv = torch.tensor(out_degree_inv, dtype=torch.float)

            edge_count = [pair[str(senders[i]) + '-' + str(receivers[i])] for i in range(len(senders))]
            edge_count = torch.tensor(edge_count, dtype=torch.float)

            edge_index = torch.tensor([senders, receivers], dtype=torch.long)
            x = torch.tensor(x, dtype=torch.long)
            y = torch.tensor([y], dtype=torch.long)
            userid = torch.tensor([userid], dtype=torch.long)
            num_count = torch.tensor(num_count, dtype=torch.float)
            sequence = torch.tensor(sequence, dtype=torch.long)
            sequence_len = torch.tensor([len(sequence)], dtype=torch.long)
            session_graph = Data(x=x, y=y, num_count=num_count,
                                 edge_index=edge_index, edge_count=edge_count,
                                 sequence=sequence, sequence_len=sequence_len,
                                 in_degree_inv=in_degree_inv, out_degree_inv=out_degree_inv, userid=userid)
            data_list.append(session_graph)
            
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
