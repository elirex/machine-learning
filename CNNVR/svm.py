#!/usr/bin/env python
from data.data_utils import load_CIFAR10
import numpy as np
import matplotlib.pyplot as plt
import time


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
CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
NUM_CLASSES = len(CLASSES)
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
print('Test data shape:', test_dataset.shape) # Preprocessing: subtract the mean image

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
from classifiers.linear_svm import *

# Generate a random SVM weight matrix of small numbers
W = np.random.randn(10, 3073) * 0.0001
# loss, grad = svm_loss_naive(W, train_dataset, train_labels, 0.00001)
# print('loss: {0:f}'.format(loss))


# Once you've implemented the gradient, recompute it with the code below
# and gradient check it with the function we provieded for you

# Compute the loss and its gradient as W.
# loss, grad = svm_loss_naive(W, train_dataset, train_labels, 0.0)
# print('Reg: 0.0, loss: {0:f}'.format(loss))

# Numerically compute the gradient along serveral randomly chosen dimenions,
# and compare them with your analyticially computed gradient. The numbers
# should match almost exactly along all dimensions.
from gradient_check import grad_check_sparse
# f = lambda w: svm_loss_naive(W, train_dataset, train_labels, 0.0)[0]
# grad_numerical = grad_check_sparse(f, W, grad, 10)

# Next implement the fuction svm_loss_vectorized; for now only compute the loss;
# we will implement the gradient in a moment.
# start_time = time.time()
# loss_naive, grad_naive = svm_loss_naive(W, train_dataset, train_labels, 0.00001)
# end_time = time.time()
# print('Naive loss: {0:e} computed in {1:f}s'.format(loss_naive, end_time - start_time))


# start_time = time.time()
# loss_vectorized, grad_vectorized = svm_loss_vectorized(W, train_dataset, train_labels, 0.00001)
# end_time = time.time()
# print('Vectorized loss: {0:e} computed in {1:f}s'.format(loss_vectorized, end_time - start_time))

# The losses should match but your vectorized implementation should be much faster.
# print('Difference: {0:f}'.format(loss_naive - loss_vectorized))


# Stochastic Gradient Descent
from classifiers import LinearSVM
svm = LinearSVM()
start_time = time.time()
loss_hist = svm.train(train_dataset, train_labels, learning_rate=1e-7,
        reg=5e4, num_iters=1500, verbose=True)
end_time = time.time()
print('Execute time:{0:f}'.format(end_time - start_time))

# A useful debugging strategy is to plot the loss as a function of
# iteration number:
# plt.plot(loss_hist)
# plt.xlabel('Iteration number')
# plt.ylabel('Loss value')
# plt.show()

# Write the LinearSVM.predict function and evalute the performance on both
# the training and validation set
y_train_pred = svm.predict(train_dataset)
print('Training accuracy: {0:f}'.format(np.mean(train_labels == y_train_pred)))
y_val_pred = svm.predict(valid_dataset)
print('Validation accurcy: {0:f}'.format(np.mean(valid_labels == y_val_pred)))


# Use the validation set to tune hyperparameters (regularization strength and learning rate)
learning_reates = [1e-7, 2e-7, 3e-7, 5e-7, 8e-7]
regularization_strengths = [1e4, 2e4, 3e4, 4e4, 5e4, 5e4, 7e4, 8e4, 1e5]

# Result is dictionary mapping tuples of the from (learning_rate, 
# regularization_strength) to tuples of the form (training_accuracy, validation_accuracy).
results = {}
best_val = -1 # The highest validation accuracy.
best_svm = None # The LinearSVM object that achieved the heighest validation rate.

iters = 2000
for lr in learning_reates:
    for rs in regularization_strengths:
        svm = LinearSVM()
        svm.train(train_dataset, train_labels, learning_rate=lr, 
                reg=rs, num_iters=iters)

        y_train_pred = svm.predict(train_dataset)
        acc_train = np.mean(train_labels == y_train_pred)
        y_val_pred = svm.predict(valid_dataset)
        acc_val = np.mean(valid_labels == y_val_pred)

        results[(lr, rs)] = (acc_train, acc_val)

        if best_val < acc_val:
            best_val = acc_val
            best_svm = svm

# Show results
for lr, rs in sorted(results):
    train_acc, val_acc = results[(lr, rs)]
    print('learning_rate: {0:e}, regularization_strengths: {1:e}, \ntrain_accuracy: {2:f}, validation_accuracy: {3:f}'.format(lr, rs, train_acc, val_acc))

print('The best validation accuracy achieved during cross-validation: {0:f}'.format(best_val))



# Show the cross-validation results
import math
x_scatter = [math.log10(x[0]) for x in results]
y_scatter = [math.log10(x[1]) for x in results]

# Training accuracy
sz = [results[x][0] * 1500 for x in results] # Default size of markers is 20
plt.subplot(1, 2, 1)
plt.scatter(x_scatter, y_scatter, sz)
plt.xlabel('Log learning rate')
plt.ylabel('Log regularization strength')
plt.title('CIFAR-10 training accuracy')

# Validation accuracy
sz = [results[x][1] * 1500 for x in results] # Default size of markers is 20
plt.subplot(1, 2, 2)
plt.scatter(x_scatter, y_scatter, sz)
plt.xlabel('Log learning rate')
plt.ylabel('Log regularization strength')
plt.title('CIFAR-10 validation accuracy')

plt.show()


# Evaluate the best svm on test set
y_test_pred = best_svm.predict(test_dataset)
test_accuracy = np.mean(test_labels == y_test_pred)
print('Linear SVM on raw pixels final test set accuracy: {0:f}'.format(test_accuracy))

# Show the learned weights for each class.
w = best_svm.W[:, :-1] # Strip out the bias
w = w.reshape(10, 32, 32, 3)
w_min, w_max = np.min(w), np.max(w)
for i in range(NUM_CLASSES):
    plt.subplot(2, 5, i + 1)
    wimg = 255.0 * (w[i].squeeze() - w_min) / (w_max - w_min)
    plt.imshow(wimg.astype['uint8'])
    plt.axis('off')
    plt.title(CLASSES[i])

