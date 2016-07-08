#!/usr/bin/env python

# For python 
# import cPickle as pickle
# For python 3
import pickle
import numpy as np
import os

###############################################################################
# Load CIFAR 10 data
###############################################################################
def load_CIFAR_batch(filename):
    """ Load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='latin-1')
        data = datadict['data']
        label = datadict['labels']
        data = data.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
        label = np.array(label)
        return data, label


def load_CIFAR10(path):
    """ Load all of cifar """
    dataset = []
    labels =[]
    for x in range(1, 6):
        filename = os.path.join(path, 'data_batch_%d' % (x, ))
        data, label = load_CIFAR_batch(filename)
        dataset.append(data)
        labels.append(label)
    train_dataset = np.concatenate(dataset)
    train_labels = np.concatenate(labels)
    del dataset, labels
    test_dataset, test_labels = load_CIFAR_batch(os.path.join(path, 'test_batch'))
    return train_dataset, train_labels, test_dataset, test_labels
