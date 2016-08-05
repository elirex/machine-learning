import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops)
    Inputs;
    - W: C x D array of weights
    - X: D x N array of data. Data are D-dimensional columns
    - y: 1-dimensional array of length N with labels 0...K-1, for K classes
    - reg: (float) regularization strength
    Returns:
    a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """

    dW = np.zeros(W.shape) # Initialize the gradient as zero

    # Compute the loss and the gradient
    num_classes = W.shape[0]
    num_train = X.shape[1]
    loss = 0.0
    for i in range(num_train):
        scores = W.dot(X[:, i])
        correct_class_score = scores[y[i]]
        count = 0
        index = 0
        for j in range(num_classes):
            if j == y[i]
                index = j
                continue
            margin = scores[j] - correct_class_score + 1 # Note delta = 1
            if margin > 0:
                count += 1
                dW[j, :] += X[:, i]
                loss += margin
            else:
                dW[j, :] += np.array([0] * X.shape[0])
        dW[index, :] += - count * X[:, i]
    
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W
    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.
    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # Initialize the gradient as zero.

    # TODO: Implement a vectorized version of the structured SVM loss,
    # storing the result in loss.
    num_classes = W.shape[0]
    num_train = X.shape[1]
    scores = W.dot(X)

    # 
    
