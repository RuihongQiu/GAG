# -*- coding: utf-8 -*-
"""
Created on 23/9/2019
@author: RuihongQiu
"""

import numpy as np
import torch
import os
from dataset import MultiSessionsGraph
from torch_geometric.data import DataLoader
from train import forward_entropy, forward_cross_entropy, forward_wass


def random_on_union(current_res, current_win, win_size, p=None):
    # R' = R U R^{new}
    uni_x = current_res[0] + current_win[0]
    uni_y = current_res[1] + current_win[1]
    uni_user = current_res[2] + current_win[2]
    
    # random sampling on the union set
    res_index = [i for i in range(len(uni_x))]
    sampled_index = np.random.choice(res_index, win_size, replace=False, p=p)
    sampled_x = np.array(uni_x)[sampled_index].tolist()
    sampled_y = np.array(uni_y)[sampled_index].tolist()
    sampled_user = np.array(uni_user)[sampled_index].tolist()
    return sampled_x, sampled_y, sampled_user


def random_on_new(current_win, win_size, p=None):
    # random sampling on the new set
    res_index = [i for i in range(len(current_win[0]))]
    sampled_index = np.random.choice(res_index, win_size, replace=False, p=p)
    sampled_x = np.array(current_win[0])[sampled_index].tolist()
    sampled_y = np.array(current_win[1])[sampled_index].tolist()
    sampled_user = np.array(current_win[2])[sampled_index].tolist()
    return sampled_x, sampled_y, sampled_user


def fix_new(current_win, win_size, max_item, max_user):
    # random sampling on the old items and users while must include all new items and users
    sampled_x, sampled_y, sampled_user = [], [], []
    deleted_index = []
    
    for i in range(len(current_win[0])):
        if max(current_win[0][i]) > max_item or current_win[1][i] > max_item or current_win[2][i] > max_user:
            sampled_x.append(current_win[0][i])
            sampled_y.append(current_win[1][i])
            sampled_user.append(current_win[2][i])
            win_size -= 1
            deleted_index.append(i)
    
    left_win = tuple(np.delete(data, deleted_index).tolist() for data in current_win)
    
    return sampled_x, sampled_y, sampled_user, left_win, win_size


def fix_new_random_on_new(current_win, win_size, max_item, max_user):
    sampled_x, sampled_y, sampled_user, left_win, win_size = fix_new(current_win, win_size, max_item, max_user)
    sampled_old = random_on_new(left_win, win_size)
    
    return sampled_x + sampled_old[0], sampled_y + sampled_old[1], sampled_user + sampled_old[2]


def fix_new_random_on_union(current_res, current_win, win_size, max_item, max_user):
    sampled_x, sampled_y, sampled_user, left_win, win_size = fix_new(current_win, win_size, max_item, max_user)
    sampled_old = random_on_union(current_res, left_win, win_size)

    return sampled_x + sampled_old[0], sampled_y + sampled_old[1], sampled_user + sampled_old[2]


def entropy_on_union(cur_dir, now, opt, model, device, current_res, current_win, win_size, ent='entropy'):
    # R' = R U R^{new}
    uni_x = current_res[0] + current_win[0]
    uni_y = current_res[1] + current_win[1]
    uni_user = current_res[2] + current_win[2]
    uni_data = (uni_x, uni_y, uni_user)

    uni_dataset = MultiSessionsGraph(cur_dir + '/../datasets/' + opt.dataset,
                                     phrase='uni' + now,
                                     sampled_data=uni_data)
    uni_loader = DataLoader(uni_dataset, batch_size=opt.batch_size, shuffle=False)
    
    with torch.no_grad():
        if ent == 'entropy':
            pro = forward_entropy(model, uni_loader, device, max(max(max(current_win[0])), max(current_win[1])))
        elif ent == 'cross':
            pro = forward_cross_entropy(model, uni_loader, device, max(max(max(current_win[0])), max(current_win[1])))
        elif ent == 'wass':
            pro = forward_wass(model, uni_loader, device, max(max(max(current_win[0])), max(current_win[1])))

    os.remove('../datasets/' + opt.dataset + '/processed/uni' + now + '.pt')
    
    return random_on_union(current_res, current_win, win_size, p=pro)


def fix_new_entropy_on_union(cur_dir, now, opt, model, device, current_res, current_win, win_size, max_item, max_user,
                             ent='entropy'):
    sampled_x, sampled_y, sampled_user, left_win, left_win_size = fix_new(current_win, win_size, max_item, max_user)
    if left_win_size > 0:
        sampled_old = entropy_on_union(cur_dir, now, opt, model, device, current_res, left_win, left_win_size, ent=ent)
    
        return sampled_x + sampled_old[0], sampled_y + sampled_old[1], sampled_user + sampled_old[2]
    else:
        return entropy_on_new(cur_dir, now, opt, model, device, (sampled_x, sampled_y, sampled_user), win_size)


def entropy_on_new(cur_dir, now, opt, model, device, current_win, win_size):
    new_dataset = MultiSessionsGraph(cur_dir + '/../datasets/' + opt.dataset,
                                     phrase='new' + now,
                                     sampled_data=current_win)
    new_loader = DataLoader(new_dataset, batch_size=opt.batch_size, shuffle=False)

    with torch.no_grad():
        pro = forward_entropy(model, new_loader, device, max(max(max(current_win[0])), max(current_win[1])))
    
    os.remove('../datasets/' + opt.dataset + '/processed/new' + now + '.pt')
    
    return random_on_new(current_win, win_size, p=pro)


def fix_new_entropy_on_new(cur_dir, now, opt, model, device, current_win, win_size, max_item, max_user):
    sampled_x, sampled_y, sampled_user, left_win, win_size = fix_new(current_win, win_size, max_item, max_user)
    sampled_old = entropy_on_new(cur_dir, now, opt, model, device, left_win, win_size)

    return sampled_x + sampled_old[0], sampled_y + sampled_old[1], sampled_user + sampled_old[2]

