#!/usr/bin/env python
from data.data_utils import load_CIFAR10
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['figure.figsize'] = (10.0, 8.0) # Set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


CIFA_PATH = 'data/cifar-10'
train_dataset, train_labels, test_dataset, test_labels = load_CIFAR10(CIFA_PATH)

# As a sanity check, we print out the size of the training and test data.
print('Training data shape:', train_dataset.shape)
print('Training labels shape:', train_labels.shape)
print('Test data shape:', test_dataset.shape)
print('Test labels shape:', test_labels.shape)


# Visualize some example from the dataset.
# CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# NUM_CLASSES = len(CLASSES)
# samples_per_class = 7
# 
# for y, cls in enumerate(CLASSES):
#     idxs = np.flatnonzero(train_labels == y)
#     idxs = np.random.choice(idxs, samples_per_class, replace=False)
#     for i, idx in enumerate(idxs):
#         plt_idx = i * NUM_CLASSES + y + 1
#         plt.subplot(samples_per_class, NUM_CLASSES, plt_idx)
#         plt.imshow(train_dataset[idx].astype('uint8'))
#         plt.axis('off')
#         if i == 0:
#             plt.title(cls)
# plt.show()


# Subsample the data for more efficient code execution in this exercise.
NUM_TRAINING = 49000
NUM_VALIDATION = 1000
NUM_TEST = 1000

# Our validation set will be NUM_VALIDATION points from the original training set.
mask = range(NUM_TRAINING, NUM_TRAINING + NUM_VALIDATION)
valid_dataset = train_dataset[mask]
valid_labels = train_labels[mask]

# Our training set will be the first NUM_TRAIN points from the original training set.
mask = range(NUM_TRAINING)
train_dataset = train_dataset[mask]
train_labels = train_labels[mask]


# We use the first NUM_TEST points of the original test set as our test set.
mask = range(NUM_TEST)
test_dataset = test_dataset[mask]
test_labels = test_labels[mask]

print('Train data shape:', train_dataset.shape)
print('Train labels shape:', train_labels.shape)
print('Validation data shape:', valid_dataset.shape)
print('Validation labels shape:', valid_labels.shape)
print('Test data shape:', test_dataset.shape)
print('Test labels shape:', test_labels.shape)


# Preprocessing: reshape the image data into rows.
train_dataset = np.reshape(train_dataset, (train_dataset.shape[0], -1))
valid_dataset = np.reshape(valid_dataset, (valid_dataset.shape[0], -1))
test_dataset = np.reshape(test_dataset, (test_dataset.shape[0], -1))

# As a sanity check, print out the shapes of the data
print('Training data shape:', train_dataset.shape)
print('Validation data shape:', valid_dataset.shape)
print('Test data shape:', test_dataset.shape)

# Preprocessing: subtract the mean image
# First: compute the image mean based on the training data
mean_image = np.mean(train_dataset, axis=0)
print(mean_image[:10]) # Prin a few of the elements
plt.figure(figsize=(4, 4))
plt.imshow(mean_image.reshape((32, 32, 3)).astype('uint8')) # Visualize the mean image

# Second: subtract the mean image from train and test data
train_dataset -= mean_image
valid_dataset -= mean_image
test_dataset -= mean_image

# Third: append the bias dimension of ones (i.e. bias trick) so that our SVM
# only has to worry about optimizing a single weight matrix W. Also, let transform both data matrices so that each image is a column,
train_dataset = np.hstack([train_dataset, np.ones((train_dataset.shape[0], 1))]).T
valid_dataset = np.hstack([valid_dataset, np.ones((valid_dataset.shape[0], 1))]).T
test_dataset = np.hstack([test_dataset, np.ones((test_dataset.shape[0], 1))]).T

print('Training data sahpe:', train_dataset.shape)
print('Validation data shape:', valid_dataset.shape)
print('Test data shape:', test_dataset.shape)


# Evaluate the naive implementation of the loss we provided for you:
from classifiers.linear_svm import svm_loss_naive

# Generate a random SVM weight matrix of small numbers
W = np.random.randn(10, 3073) * 0.0001
loss, grad = svm_loss_naive(W, train_dataset, train_labels, 0.00001)
print('loss: {0:f}'.format(loss))


# Once you've implemented the gradient, recompute it with the code below
# and gradient check it with the function we provieded for you

# Compute the loss and its gradient as W.
loss, grad = svm_loss_naive(W, train_dataset, train_labels, 0.0)
print('Reg: 0.0, loss: {0:f}'.format(loss))

# Numerically compute the gradient along serveral randomly chosen dimenions,
# and compare them with your analyticially computed gradient. The numbers
# should match almost exactly along all dimensions.
from gradient_check import grad_check_sparse
f = lambda w: svm_loss_naive(W, train_dataset, train_labels, 0.0)[0]
grad_numerical = grad_check_sparse(f, W, grad, 10)

