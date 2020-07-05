# -*- coding: utf-8 -*-
"""
Created on 19/9/2019
@author: LeiGuo, RuihongQiu
"""

import numpy as np


class Reservoir(object):
    def __init__(self, train, size_denominator):
        super(Reservoir, self).__init__()
        self.r_size = len(train[0]) / size_denominator
        self.t = 0
        self.data = ([], [], [])

    def add(self, x, y, u):  # one list represents one sample
        # global t
        if self.t < self.r_size:
            self.data[0].append(x)
            self.data[1].append(y)
            self.data[2].append(u)
        else:
            p = self.r_size / self.t
            s = False
            random = np.random.rand()
            if random <= p:
                s = True
            if s:
                random = np.random.rand()
                index = int(random * (len(self.data[0]) - 1))
                self.data[0][index] = x
                self.data[1][index] = y
                self.data[2][index] = u
        self.t += 1

    def update(self, data):
        for index in range(len(data[0])):
            x = data[0][index]
            y = data[1][index]
            user = data[2][index]
            self.add(x, y, user)
