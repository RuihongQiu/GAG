# -*- coding: utf-8 -*-
"""
Created on 17/9/2019
@author: RuihongQiu
"""

import argparse
import logging
import time
from tqdm import tqdm
from model import GNNModel
from train import forward
from torch.utils.tensorboard import SummaryWriter
from se_data_process import load_data_valid, load_testdata
from reservoir import Reservoir
from sampling import *

# Logger configuration
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(message)s')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='gowalla', help='dataset name: gowalla/lastfm')
parser.add_argument('--batch_size', type=int, default=100, help='input batch size')
parser.add_argument('--hidden_size', type=int, default=200, help='hidden state size')
parser.add_argument('--epoch', type=int, default=4, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.003, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=1.0, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--u', type=int, default=1, help='the number of layer with u')
parser.add_argument('--res_size', type=int, default=100, help='the denominator of the reservoir size')
parser.add_argument('--win_size', type=int, default=1, help='the denominator of the window size')
opt = parser.parse_args()
logging.warning(opt)


def main():
    assert opt.dataset in ['gowalla', 'lastfm']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    cur_dir = os.getcwd()
    
    train_dataset = MultiSessionsGraph(cur_dir + '/../datasets/' + opt.dataset, phrase='train')
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    train_for_res, _ = load_data_valid(
        os.path.expanduser(os.path.normpath(cur_dir + '/../datasets/' + opt.dataset + '/raw/train.txt.csv')), 0)
    max_train_item = max(max(max(train_for_res[0])), max(train_for_res[1]))
    max_train_user = max(train_for_res[2])
    
    test_dataset = MultiSessionsGraph(cur_dir + '/../datasets/' + opt.dataset, phrase='test1')
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)
    test_for_res = load_testdata(
        os.path.expanduser(os.path.normpath(cur_dir + '/../datasets/' + opt.dataset + '/raw/test1.txt.csv')))
    max_item = max(max(max(test_for_res[0])), max(test_for_res[1]))
    max_user = max(test_for_res[2])
    pre_max_item = max_train_item
    pre_max_user = max_train_user
    
    log_dir = cur_dir + '/../log/' + str(opt.dataset) + '/paper200/' + str(
        opt) + '_fix_new_entropy(rank)_on_union+' + str(opt.u) + 'tanh*u_AGCN***GAG-win' + str(opt.win_size) \
              + '***concat3_linear_tanh_in_e2s_' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.warning('logging to {}'.format(log_dir))
    writer = SummaryWriter(log_dir)
    
    if opt.dataset == 'gowalla':
        n_item = 30000
        n_user = 33005
    else:
        n_item = 10000
        n_user = 984
    
    model = GNNModel(hidden_size=opt.hidden_size, n_item=n_item, n_user=n_user, u=opt.u).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 3], gamma=opt.lr_dc)
    
    logging.warning(model)
    
    # offline training on 'train' and test on 'test1'
    logging.warning('*********Begin offline training*********')
    updates_per_epoch = len(train_loader)
    updates_count = 0
    for train_epoch in tqdm(range(opt.epoch)):
        forward(model, train_loader, device, writer, train_epoch, optimizer=optimizer,
                train_flag=True, max_item_id=max_train_item, last_update=updates_count)
        scheduler.step()
        updates_count += updates_per_epoch
        with torch.no_grad():
            forward(model, test_loader, device, writer, train_epoch, train_flag=False, max_item_id=max_item)
    
    # reservoir construction with 'train'
    logging.warning('*********Constructing the reservoir with offline training data*********')
    res = Reservoir(train_for_res, opt.res_size)
    res.update(train_for_res)
    
    # test and online training on 'test2~5'
    logging.warning('*********Begin online training*********')
    now = time.asctime()
    for test_epoch in tqdm(range(1, 6)):
        if test_epoch != 1:
            test_dataset = MultiSessionsGraph(cur_dir + '/../datasets/' + opt.dataset, phrase='test' + str(test_epoch))
            test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)
            
            test_for_res = load_testdata(
                os.path.expanduser(os.path.normpath(
                    cur_dir + '/../datasets/' + opt.dataset + '/raw/test' + str(test_epoch) + '.txt.csv')))
            pre_max_item = max_item
            pre_max_user = max_user
            max_item = max(max(max(test_for_res[0])), max(test_for_res[1]))
            max_user = max(test_for_res[2])
            
            # test on the current test set
            # no need to test on test1 because it's done in the online training part
            # epoch + 10 is a number only for the visualization convenience
            with torch.no_grad():
                forward(model, test_loader, device, writer, test_epoch + 10,
                        train_flag=False, max_item_id=max_item)
        
        # reservoir sampling
        sampled_data = fix_new_entropy_on_union(cur_dir, now, opt, model, device, res.data, test_for_res,
                                                len(test_for_res[0]) // opt.win_size, pre_max_item, pre_max_user,
                                                ent='wass')
        
        # cast the sampled set to dataset
        sampled_dataset = MultiSessionsGraph(cur_dir + '/../datasets/' + opt.dataset,
                                             phrase='sampled' + now,
                                             sampled_data=sampled_data)
        sampled_loader = DataLoader(sampled_dataset, batch_size=opt.batch_size, shuffle=True)
        
        # update with the sampled set
        forward(model, sampled_loader, device, writer, test_epoch + opt.epoch, optimizer=optimizer,
                train_flag=True, max_item_id=max_item, last_update=updates_count)
        
        updates_count += len(test_loader)
        
        scheduler.step()
        
        res.update(test_for_res)
        os.remove('../datasets/' + opt.dataset + '/processed/sampled' + now + '.pt')


if __name__ == '__main__':
    main()
