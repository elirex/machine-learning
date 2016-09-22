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
# start_time = time.time()
# loss_naive, grad_naive = softmax_loss_naive(W, train_dataset, train_labels, 0.00001)
# end_time = time.time()
# print('Naive loss: {0:e}, computed in {1:f}s'.format(loss_naive, end_time - start_time))

# As a rough sanity check, our loss should be something close to -log(0.1).
# print('Sanity check: {0:f}'.format(-np.log(0.1)))

# Use numeric grdient checking as a debugging tool.
from gradient_check import grad_check_sparse
# f = lambda w: softmax_loss_naive(W, train_dataset, train_labels, 0.0)[0]
# grad_numerical = grad_check_sparse(f, W, grad_naive, 10)


# start_time = time.time()
# loss_vectorized, grad_vectorized = softmax_loss_vectorized(W, train_dataset, train_labels, 0.00001)
# end_time = time.time()
# print('Vectorized loss: {0:e}, computed in {1:f}s'.format(loss_vectorized, end_time - start_time))

# Use the Frobenius norm to compare the two versions of the gradient.
# grad_difference = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')
# print('Loss difference: {0:f}'.format(np.abs(loss_naive - loss_vectorized)))
# print('Gradient difference: {0:f}'.format(grad_difference))




# Use the validation set to tune hyperparameters (regularization strength 
# and learning rate). You should experiment with different ranges for the
# learning rates and regularization strengths; if you are careful you 
# should be able to get a classification accuracy of over 0.35 on the 
# validation set.
from classifiers import Softmax

results = {}
best_val = -1
best_softmax = None
learning_rates = np.logspace(-10, 10, 10)
regularization_strengths = np.logspace(-3, 6, 10)

for rate in learning_rates:
    for strength in regularization_strengths:
        softmax = Softmax()
        softmax.train2(train_dataset, train_labels, learning_rate=rate,
                reg=strength, num_iters=1500, verbose=False)
        num_samples_train = train_dataset.shape[0]
        num_sample_valid = valid_dataset[0]
        learning_accuracy = np.mean(softmax.predict2(train_dataset) == train_labels)
        validation_accuracy = np.mean(softmax.predict2(valid_dataset) == valid_labels)

        if validation_accuracy > best_val:
            best_val = validation_accuracy
            best_softmax = softmax
        
        results[(rate, strength)] = (learning_accuracy, validation_accuracy)


# Print the results
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr {0:e} reg {1:e} train accuracy: {2:f} val accuracy: {3:f}'.format(lr, reg, train_accuracy, val_accuracy))

print('Best validation accuracy achieved during cross-validation: {0:f}'.format(best_val))

# Evaluate on test set
# Evaluate the best softmax on test set
test_labels_pred = best_softmax.predict2(test_dataset)
test_accuracy = np.mean(test_labels == test_labels_pred)
print('Softmax on raw pixels final test set accuracy: {0:f}'.format(test_accuracy))

# Visualize the leaned weights for each class
w = best_softmax.W[:-1, :] # Strip out the bias
w = w.reshape(32, 32, 3, 10)

w_min, w_max = np.min(w), np.max(w)

CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
        'ship', 'truck']
for i in range(10):
    plt.subplot(2, 5, i + 1)

    # Rescale the weights to be between 0 and 255
    wimg = 255.0 * (w[:, :, :, i].sqeeze() - w_min) / (w_max - w_min)
    plt.imshow(wimg.astype('uint8'))
    plt.asis('off')
    plt.title(CLASSES[i])

