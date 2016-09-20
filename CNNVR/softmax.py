#!/usr/bin/env python
from data.data_utils import load_CIFAR10
import numpy as np
import matplotlib.pyplot as plt
import time


plt.rcParams['figure.figsize'] = (10.0, 8.0) # Set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def get_CIFAR10_data(num_training=49000, num_validation=1000, 
        num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the linear classifier. These are the same steps as we used for
    the SVM, but condensed to single function.
    """

    # Load the raw CIFAR-10 data
    CIFA_PATH = 'data/cifar-10'
    train_dataset, train_labels, test_dataset, test_labels = load_CIFAR10(CIFA_PATH)

    # subsample the data
    mask = range(num_training, num_training + num_validation)
    valid_dataset = train_dataset[mask]
    valid_labels = train_labels[mask]
    mask = range(num_training)
    train_dataset = train_dataset[mask]
    train_labels = train_labels[mask]
    mask = range(num_test)
    test_dataset = test_dataset[mask]
    test_labels = test_labels[mask]

    # Preprocessing: reshape the image data into rows
    train_dataset = np.reshape(train_dataset, (train_dataset.shape[0], -1))
    valid_dataset = np.reshape(valid_dataset, (valid_dataset.shape[0], -1))
    test_dataset = np.reshape(test_dataset, (test_dataset.shape[0], -1))

    # Normalize the data: subtract the mean image
    mean_image = np.mean(train_dataset, axis = 0)
    train_dataset -= mean_image
    valid_dataset -= mean_image
    test_dataset -= mean_image

    # Add bias dimension and transform into columns
    train_dataset = np.hstack([train_dataset, np.ones((train_dataset.shape[0], 1))])
    valid_dataset = np.hstack([valid_dataset, np.ones((valid_dataset.shape[0], 1))])
    test_dataset = np.hstack([test_dataset, np.ones((test_dataset.shape[0], 1))])

    return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels

# Invoke the above function ot get our data.
train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = get_CIFAR10_data()

print('Train dataset shape:', train_dataset.shape)
print('Train lables shape:', train_labels.shape)
print('Validation dataset shape:', valid_dataset.shape)
print('Valid labels shape:', valid_labels.shape)
print('Test dataset shape:', test_dataset.shape)
print('Test labels shape:', test_labels.shape)


# First implement the naive softmax loss function with nested loops.
from classifiers.softmax import *

# Generate a random softmax weight matrix and use it to compute the loss.
W = np.random.randn(3073, 10) * 0.0001
start_time = time.time()
loss_naive, grad_naive = softmax_loss_naive(W, train_dataset, train_labels, 0.00001)
end_time = time.time()
print('Naive loss: {0:e}, computed in {1:f}s'.format(loss_naive, end_time - start_time))

# As a rough sanity check, our loss should be something close to -log(0.1).
print('Sanity check: {0:f}'.format(-np.log(0.1)))

# Use numeric grdient checking as a debugging tool.
from gradient_check import grad_check_sparse
f = lambda w: softmax_loss_naive(W, train_dataset, train_labels, 0.0)[0]
grad_numerical = grad_check_sparse(f, W, grad_naive, 10)


start_time = time.time()
loss_vectorized, grad_vectorized = softmax_loss_vectorized(W, train_dataset, train_labels, 0.00001)
end_time = time.time()
print('Vectorized loss: {0:e}, computed in {1:f}s'.format(loss_vectorized, end_time - start_time))

# Use the Frobenius norm to compare the two versions of the gradient.
grad_difference = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')
print('Loss difference: {0:f}'.format(np.abs(loss_naive - loss_vectorized)))
print('Gradient difference: {0:f}'.format(grad_difference))


