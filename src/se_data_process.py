# -*- coding: utf-8 -*-
"""
Created on 17/9/2019
@author: LeiGuo, RuihongQiu
"""

import pickle
import numpy


def prepare_data(seqs, labels):
    """Create the matrices from the datasets.
    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.
    if maxlen is set, we will cut all sequence to this maximum
    lenght.
    This swap the axis!
    """
    lengths = [len(s) for s in seqs]
    n_samples = len(seqs)
    maxlen = numpy.max(lengths)

    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    x_mask = numpy.ones((maxlen, n_samples)).astype(theano.config.floatX)
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s

    x_mask *= (1 - (x == 0))
    return x, x_mask, labels


def load_data(path, valid_portion=0.1, maxlen=19, sort_by_len=False):
    """Loads the dataset
    :type path: String
    :param path: The path to the dataset (here RSC2015)
    :type n_items: int
    :param n_items: The number of items.
    :type valid_portion: float
    :param valid_portion: The proportion of the full train set used for
        the validation set.
    :type maxlen: None or positive int
    :param maxlen: the max sequence length we use in the train/valid set.
    :type sort_by_len: bool
    :name sort_by_len: Sort by the sequence lenght for the train,
        valid and test set. This allow faster execution as it cause
        less padding per minibatch. Another mechanism must be used to
        shuffle the train set at each epoch.
    """
    path_train_data = path+'train_final.csv'
    path_test_data = path+'test_final.csv'

    f1 = open(path_train_data, 'rb')
    train_set = pickle.load(f1)
    f1.close()

    f2 = open(path_test_data, 'rb')
    test_set = pickle.load(f2)
    f2.close()

    if maxlen:
        new_train_set_x = []
        new_train_set_y = []
        new_train_set_u = []
        for x, y, u in list(zip(train_set[0], train_set[1],train_set[2])):
            if len(x) < maxlen:
                new_train_set_x.append(x)
                new_train_set_y.append(y)
                new_train_set_u.append(u)
            else:
                new_train_set_x.append(x[:maxlen])
                new_train_set_y.append(y)
                new_train_set_u.append(u)
        train_set = (new_train_set_x, new_train_set_y, new_train_set_u)
        del new_train_set_x, new_train_set_y, new_train_set_u

        new_test_set_x = []
        new_test_set_y = []
        new_test_set_u = []
        for xx, yy, uu in zip(test_set[0], test_set[1], test_set[2]):
            if len(xx) < maxlen:
                new_test_set_x.append(xx)
                new_test_set_y.append(yy)
                new_test_set_u.append(uu)
            else:
                new_test_set_x.append(xx[:maxlen])
                new_test_set_y.append(yy)
                new_test_set_u.append(uu)
        test_set = (new_test_set_x, new_test_set_y, new_test_set_u)
        del new_test_set_x, new_test_set_y, new_test_set_u

    # split training set into validation set
    train_set_x, train_set_y, train_set_u = train_set
    n_samples = len(train_set_x)
    sidx = numpy.arange(n_samples, dtype='int32')
    numpy.random.shuffle(sidx)
    n_train = int(numpy.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    valid_set_u = [train_set_u[s] for s in sidx[n_train:]]

    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]
    train_set_u = [train_set_u[s] for s in sidx[:n_train]]

    train_set = (train_set_x, train_set_y, train_set_u)
    valid_set = (valid_set_x, valid_set_y, valid_set_u)

    test_set_x, test_set_y, test_set_u = test_set
    valid_set_x, valid_set_y, valid_set_u = valid_set
    train_set_x, train_set_y, train_set_u = train_set

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(test_set_x)
        test_set_x = [test_set_x[i] for i in sorted_index]
        test_set_y = [test_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(valid_set_x)
        valid_set_x = [valid_set_x[i] for i in sorted_index]
        valid_set_y = [valid_set_y[i] for i in sorted_index]

    train = (train_set_x, train_set_y, train_set_u)
    valid = (valid_set_x, valid_set_y, valid_set_u)
    test = (test_set_x, test_set_y, test_set_u)

    return train, valid, test


def load_traindata(trainFile, validFile, valid_portion=0.1, maxlen=19, sort_by_len=False):
    """Loads the dataset
    :type path: String
    :param path: The path to the dataset (here RSC2015)
    :type n_items: int
    :param n_items: The number of items.
    :type valid_portion: float
    :param valid_portion: The proportion of the full train set used for
        the validation set.
    :type maxlen: None or positive int
    :param maxlen: the max sequence length we use in the train/valid set.
    :type sort_by_len: bool
    :name sort_by_len: Sort by the sequence lenght for the train,
        valid and test set. This allow faster execution as it cause
        less padding per minibatch. Another mechanism must be used to
        shuffle the train set at each epoch.
    """

    path_train_data = trainFile
    path_test_data = validFile

    f1 = open(path_train_data, 'rb')
    train_set = pickle.load(f1)
    f1.close()

    f2 = open(path_test_data, 'rb')
    test_set = pickle.load(f2)
    f2.close()

    if maxlen:
        new_train_set_x = []
        new_train_set_y = []
        new_train_set_u = []
        for x, y, u in list(zip(train_set[0], train_set[1], train_set[2])):
            if len(x) < maxlen:
                new_train_set_x.append(x)
                new_train_set_y.append(y)
                new_train_set_u.append(u)
            else:
                new_train_set_x.append(x[:maxlen])
                new_train_set_y.append(y)
                new_train_set_u.append(u)
        train_set = (new_train_set_x, new_train_set_y, new_train_set_u)
        del new_train_set_x, new_train_set_y, new_train_set_u

        new_test_set_x = []
        new_test_set_y = []
        new_test_set_u = []
        for xx, yy, uu in zip(test_set[0], test_set[1], test_set[2]):
            if len(xx) < maxlen:
                new_test_set_x.append(xx)
                new_test_set_y.append(yy)
                new_test_set_u.append(uu)
            else:
                new_test_set_x.append(xx[:maxlen])
                new_test_set_y.append(yy)
                new_test_set_u.append(uu)
        test_set = (new_test_set_x, new_test_set_y, new_test_set_u)
        del new_test_set_x, new_test_set_y, new_test_set_u

    test_set_x, test_set_y, test_set_u = test_set
    train_set_x, train_set_y, train_set_u = train_set

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(test_set_x)
        test_set_x = [test_set_x[i] for i in sorted_index]
        test_set_y = [test_set_y[i] for i in sorted_index]

    train = (train_set_x, train_set_y, train_set_u)
    test = (test_set_x, test_set_y, test_set_u)

    return train, test


def load_testdata(testFile, maxlen=19, sort_by_len=False):
    """Loads the dataset
    :type path: String
    :param path: The path to the dataset (here RSC2015)
    :type n_items: int
    :param n_items: The number of items.
    :type valid_portion: float
    :param valid_portion: The proportion of the full train set used for
        the validation set.
    :type maxlen: None or positive int
    :param maxlen: the max sequence length we use in the train/valid set.
    :type sort_by_len: bool
    :name sort_by_len: Sort by the sequence lenght for the train,
        valid and test set. This allow faster execution as it cause
        less padding per minibatch. Another mechanism must be used to
        shuffle the train set at each epoch.
    """
    path_test_data = testFile

    f2 = open(path_test_data, 'rb')
    test_set = pickle.load(f2)
    f2.close()

    if maxlen:
        new_test_set_x = []
        new_test_set_y = []
        new_test_set_u = []
        for xx, yy, uu in zip(test_set[0], test_set[1], test_set[2]):
            if len(xx) < maxlen:
                new_test_set_x.append(xx)
                new_test_set_y.append(yy)
                new_test_set_u.append(uu)
            else:
                new_test_set_x.append(xx[:maxlen])
                new_test_set_y.append(yy)
                new_test_set_u.append(uu)
        test_set = (new_test_set_x, new_test_set_y, new_test_set_u)
        del new_test_set_x, new_test_set_y, new_test_set_u

    test_set_x, test_set_y, test_set_u = test_set

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(test_set_x)
        test_set_x = [test_set_x[i] for i in sorted_index]
        test_set_y = [test_set_y[i] for i in sorted_index]

    test = (test_set_x, test_set_y, test_set_u)

    return test


def load_data_valid(train_file, valid_portion=0.1, maxlen=19, sort_by_len=False):
    path_train_data = train_file

    f1 = open(path_train_data, 'rb')
    train_set = pickle.load(f1)
    f1.close()
    if maxlen:
        new_train_set_x = []
        new_train_set_y = []
        new_train_set_u = []
        for x, y, u in list(zip(train_set[0], train_set[1],train_set[2])):
            if len(x) < maxlen:
                new_train_set_x.append(x)
                new_train_set_y.append(y)
                new_train_set_u.append(u)
            else:
                new_train_set_x.append(x[:maxlen])
                new_train_set_y.append(y)
                new_train_set_u.append(u)
        train_set = (new_train_set_x, new_train_set_y, new_train_set_u)
        del new_train_set_x, new_train_set_y, new_train_set_u

    # split training set into validation set
    train_set_x, train_set_y, train_set_u = train_set
    n_samples = len(train_set_x)
    sidx = numpy.arange(n_samples, dtype='int32')
    numpy.random.shuffle(sidx)
    n_train = int(numpy.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    valid_set_u = [train_set_u[s] for s in sidx[n_train:]]

    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]
    train_set_u = [train_set_u[s] for s in sidx[:n_train]]

    train_set = (train_set_x, train_set_y, train_set_u)
    valid_set = (valid_set_x, valid_set_y, valid_set_u)

    valid_set_x, valid_set_y, valid_set_u = valid_set
    train_set_x, train_set_y, train_set_u = train_set

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:

        sorted_index = len_argsort(valid_set_x)
        valid_set_x = [valid_set_x[i] for i in sorted_index]
        valid_set_y = [valid_set_y[i] for i in sorted_index]

    train = (train_set_x, train_set_y, train_set_u)
    valid = (valid_set_x, valid_set_y, valid_set_u)

    return train, valid
