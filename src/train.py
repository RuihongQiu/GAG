# -*- coding: utf-8 -*-
"""
Created on 5/4/2019
@author: RuihongQiu
"""

import numpy as np
import torch
import pandas as pd
from torch.nn.functional import softmax
from torch.distributions.categorical import Categorical
from scipy.stats import wasserstein_distance


def forward(model, loader, device, writer, epoch, optimizer=None, train_flag=True, max_item_id=0, last_update=0):
    if train_flag:
        model.train()
    else:
        model.eval()
        hit20, mrr20, hit10, mrr10, hit5, mrr5, hit1, mrr1 = [], [], [], [], [], [], [], []

    mean_loss = 0.0

    for i, batch in enumerate(loader):
        if train_flag:
            optimizer.zero_grad()
        scores = model(batch.to(device), max_item_id)
        targets = batch.y - 1
        loss = model.loss_function(scores, targets)

        if train_flag:
            loss.backward()
            optimizer.step()
            writer.add_scalar('loss/train_batch_loss', loss.item(), last_update + i)
            writer.add_scalar('embedding/user_embedding', model.user_embedding.weight.mean(), last_update + i)
            writer.add_scalar('embedding/item_embedding', model.item_embedding.weight.mean(), last_update + i)
        else:
            sub_scores = scores.topk(20)[1]    # batch * top_k indices
            for score, target in zip(sub_scores.detach().cpu().numpy(), targets.detach().cpu().numpy()):
                hit20.append(np.isin(target, score))
                if len(np.where(score == target)[0]) == 0:
                    mrr20.append(0)
                else:
                    mrr20.append(1 / (np.where(score == target)[0][0] + 1))

            sub_scores = scores.topk(10)[1]  # batch * top_k indices
            for score, target in zip(sub_scores.detach().cpu().numpy(), targets.detach().cpu().numpy()):
                hit10.append(np.isin(target, score))
                if len(np.where(score == target)[0]) == 0:
                    mrr10.append(0)
                else:
                    mrr10.append(1 / (np.where(score == target)[0][0] + 1))

            sub_scores = scores.topk(5)[1]  # batch * top_k indices
            for score, target in zip(sub_scores.detach().cpu().numpy(), targets.detach().cpu().numpy()):
                hit5.append(np.isin(target, score))
                if len(np.where(score == target)[0]) == 0:
                    mrr5.append(0)
                else:
                    mrr5.append(1 / (np.where(score == target)[0][0] + 1))

            sub_scores = scores.topk(1)[1]  # batch * top_k indices
            for score, target in zip(sub_scores.detach().cpu().numpy(), targets.detach().cpu().numpy()):
                hit1.append(np.isin(target, score))
                if len(np.where(score == target)[0]) == 0:
                    mrr1.append(0)
                else:
                    mrr1.append(1 / (np.where(score == target)[0][0] + 1))

        mean_loss += loss / batch.num_graphs

    if train_flag:
        writer.add_scalar('loss/train_loss', mean_loss.item(), epoch)
    else:
        writer.add_scalar('loss/test_loss', mean_loss.item(), epoch)
        hit20 = np.mean(hit20) * 100
        mrr20 = np.mean(mrr20) * 100
        writer.add_scalar('index/hit20', hit20, epoch)
        writer.add_scalar('index/mrr20', mrr20, epoch)
        hit10 = np.mean(hit10) * 100
        mrr10 = np.mean(mrr10) * 100
        writer.add_scalar('index/hit10', hit10, epoch)
        writer.add_scalar('index/mrr10', mrr10, epoch)
        hit5 = np.mean(hit5) * 100
        mrr5 = np.mean(mrr5) * 100
        writer.add_scalar('index/hit5', hit5, epoch)
        writer.add_scalar('index/mrr5', mrr5, epoch)
        hit1 = np.mean(hit1) * 100
        mrr1 = np.mean(mrr1) * 100
        writer.add_scalar('index/hit1', hit1, epoch)
        writer.add_scalar('index/mrr1', mrr1, epoch)


def forward_entropy(model, loader, device, max_item_id=0):
    for i, batch in enumerate(loader):
        scores = softmax(model(batch.to(device), max_item_id), dim=1)
        dis_score = Categorical(scores)
        if i == 0:
            entropy = dis_score.entropy()
        else:
            entropy = torch.cat((entropy, dis_score.entropy()))
    
    # pro = softmax(entropy).cpu().detach().numpy()
    pro = entropy.cpu().detach().numpy()
    weights = np.exp((pd.Series(pro).rank() / len(pro)).values)
    return weights / np.sum(weights)
    # return pro / pro.sum()


def forward_cross_entropy(model, loader, device, max_item_id=0):
    for i, batch in enumerate(loader):
        scores = softmax(model(batch.to(device), max_item_id), dim=1)
        targets = batch.y - 1
        if i == 0:
            cross_entropy = torch.nn.functional.cross_entropy(scores, targets, reduction='none')
        else:
            cross_entropy = torch.cat((cross_entropy, torch.nn.functional.cross_entropy(scores, targets, reduction='none')))

    pro = cross_entropy.cpu().detach().numpy()
    return pro / pro.sum()


def forward_wass(model, loader, device, max_item_id=0):
    distance = []
    for i, batch in enumerate(loader):
        scores = softmax(model(batch.to(device), max_item_id), dim=1)
        targets = batch.y - 1
        targets_1hot = torch.zeros_like(scores).scatter_(1, targets.view(-1, 1), 1).cpu().numpy()
        distance += list(wasserstein_distance(score, target) for score, target in zip(scores.cpu().numpy(), targets_1hot))
        # distance += list(map(wasserstein_distance, scores.cpu().numpy(), targets_1hot))
    
    weights = np.exp((pd.Series(distance).rank() / len(distance)).values)
    return weights / np.sum(weights)
    # return distance / np.sum(distance)
