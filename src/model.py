# -*- coding: utf-8 -*-
"""
Created on 17/9/2019
@author: RuihongQiu
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from AGCN import AGCN


class Embedding2Score(nn.Module):
    def __init__(self, hidden_size):
        super(Embedding2Score, self).__init__()
        self.hidden_size = hidden_size
        self.W_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.q = nn.Linear(self.hidden_size, 1)
        self.W_3 = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.user_linear = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, node_embedding, item_embedding_table, sections, num_count, user_embedding, max_item_id, u_n_repeat):
        v_i = torch.split(node_embedding, tuple(sections.cpu().numpy()))    # split whole x back into graphs G_i
        v_n_repeat = tuple(nodes[-1].view(1, -1).repeat(nodes.shape[0], 1) for nodes in v_i)    # repeat |V|_i times for the last node embedding

        # Eq(6)
        alpha = self.q(torch.sigmoid(self.W_1(torch.cat(v_n_repeat, dim=0)) + self.W_2(node_embedding)))    # |V|_i * 1
        s_g_whole = num_count.view(-1, 1) * alpha * node_embedding    # |V|_i * hidden_size
        s_g_split = torch.split(s_g_whole, tuple(sections.cpu().numpy()))    # split whole s_g into graphs G_i
        s_g = tuple(torch.sum(embeddings, dim=0).view(1, -1) for embeddings in s_g_split)

        # Eq(7)
        v_n = tuple(nodes[-1].view(1, -1) for nodes in v_i)
        s_h = self.W_3(torch.cat((torch.cat(v_n, dim=0), torch.cat(s_g, dim=0)), dim=1))
        
        return s_h


class Embedding2ScoreWithU(nn.Module):
    def __init__(self, hidden_size):
        super(Embedding2ScoreWithU, self).__init__()
        self.hidden_size = hidden_size
        self.W_1 = nn.Linear(self.hidden_size, 1)
        self.W_2 = nn.Linear(2 * self.hidden_size + self.hidden_size, self.hidden_size)
        self.W_3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_4 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_5 = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.user_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.user_out = nn.Linear(2 * self.hidden_size, self.hidden_size)

    def forward(self, node_embedding, item_embedding_table, sections, num_count, user_embedding, max_item_id,
                u_n_repeat):
        if list(sections.size())[0] == 1:
            u_n_repeat = u_n_repeat.view(1, -1)
            node_embedding = node_embedding.view(-1, self.hidden_size)
            v_n_repeat = tuple(node_embedding.repeat(sections[0], 1))
            alpha = self.W_1(
                torch.sigmoid(self.W_2(torch.cat((v_n_repeat[0].view(1, -1), node_embedding, u_n_repeat), dim=-1))))
        else:
            v_i = torch.split(node_embedding, tuple(sections.cpu().numpy()))  # split whole x back into graphs G_i
            v_n_repeat = tuple(nodes[-1].view(1, -1).repeat(nodes.shape[0], 1) for nodes in
                               v_i)  # repeat |V|_i times for the last node embedding
        
            alpha = self.W_1(
                torch.sigmoid(self.W_2(torch.cat((torch.cat(v_n_repeat, dim=0), node_embedding, u_n_repeat), dim=-1))))
        s_g_whole = num_count.view(-1, 1) * alpha * node_embedding  # |V|_i * hidden_size
        if list(sections.size())[0] == 1:
            s_g = tuple(torch.sum(s_g_whole.view(-1, self.hidden_size), dim=0).view(1, -1))
            s_h = self.W_5(torch.cat((node_embedding, s_g[0].view(-1, self.hidden_size)), dim=-1))
        else:
            s_g_split = torch.split(s_g_whole, tuple(sections.cpu().numpy()))  # split whole s_g into graphs G_i
            s_g = tuple(torch.sum(embeddings, dim=0).view(1, -1) for embeddings in s_g_split)
        
            v_n = tuple(nodes[-1].view(1, -1) for nodes in v_i)
            stack_v_n = torch.cat(v_n, dim=0)
            s_h = self.W_5(torch.cat((stack_v_n, torch.cat(s_g, dim=0)), dim=-1))
        
        s_h += self.user_linear(user_embedding).tanh()
        return s_h


class GNNModel(nn.Module):
    """
    Args:
        hidden_size: the number of units in a hidden layer.
        n_item: the number of items in the whole item set for embedding layer.
        n_user: the number of users
    """
    def __init__(self, hidden_size, n_item, n_user=None, heads=None, u=1):
        super(GNNModel, self).__init__()
        self.hidden_size, self.n_item, self.n_user, self.heads, self.u = hidden_size, n_item, n_user, heads, u
        self.item_embedding = nn.Embedding(self.n_item, self.hidden_size)
        if self.n_user:
            self.user_embedding = nn.Embedding(self.n_user, self.hidden_size)
        if self.u > 0:
            self.gnn = []
            for i in range(self.u):
                self.gnn.append(AGCN(2 * self.hidden_size, self.hidden_size).cuda())
        else:
            self.gnn = AGCN(self.hidden_size, self.hidden_size)
        self.e2s = Embedding2ScoreWithU(self.hidden_size)
        self.loss_function = nn.CrossEntropyLoss()
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, data, max_item_id=0):
        x, edge_index, batch, edge_count, in_degree_inv, out_degree_inv, sequence, num_count, userid = \
            data.x - 1, data.edge_index, data.batch, data.edge_count, data.in_degree_inv, data.out_degree_inv,\
            data.sequence, data.num_count, data.userid - 1

        hidden = self.item_embedding(x).squeeze()
        sections = torch.bincount(batch)
        u = self.user_embedding(userid).squeeze()
        
        if self.u > 0:
            for layer in range(self.u):
                if list(sections.size())[0] == 1:
                    u_n_repeat = tuple(u.view(1, -1).repeat(sections[0], 1))
                else:
                    u_n_repeat = tuple(u.view(1, -1).repeat(times, 1) for (u, times) in zip(u, sections))
                hidden = self.gnn[layer](hidden, edge_index,
                                         [edge_count * in_degree_inv, edge_count * out_degree_inv],
                                         u=torch.cat(u_n_repeat, dim=0))
                if self.heads is not None:
                    hidden = torch.stack(hidden.chunk(self.heads, dim=-1), dim=1).mean(dim=1)
                hidden = torch.tanh(hidden)
                u = self.e2s(hidden, self.item_embedding, sections, num_count, u, max_item_id, torch.cat(u_n_repeat, dim=0))
        else:
            hidden = self.gnn(hidden, edge_index, [edge_count * in_degree_inv, edge_count * out_degree_inv], u=None)
            if self.heads is not None:
                hidden = torch.stack(hidden.chunk(self.heads, dim=-1), dim=1).mean(dim=1)
            u = self.e2s(hidden, self.item_embedding, sections, num_count, u, max_item_id)
        
        z_i_hat = torch.mm(u, self.item_embedding.weight[:max_item_id].transpose(1, 0))
        return z_i_hat
