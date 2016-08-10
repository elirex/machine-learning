import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loop).
    Inputs:
    - W: C x D array of weights.
    - X: D x N array of data. Data are D-dimensional columns.
    - y: 1-dimensional array of length N with labels 0...K-1, for K classes.
    - reg: (float) regularization strength.

    Returns:
    a tuple of:
    - loss as single float.
    - gradient with respect to weights W, an array of same size as W.
    """

    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    # Compute the softmax and its gradient using explicit loops.
    # Gat shapes
    num_classes = W.shape[0]
    num_train = W.shape[1]

    for i in range(num_train):
        # Compute vector of scores
        f_i = W.dot(X[:, i]) # In R^{num_classes}

        # Normalization trick to avoid numerical instability
        log_c = np.max(f_i)
        f_i -= log_c

        # Compute loss (and add to it, divided later)
        # L_i = - f(x_i)_{y_i} + log \ sum_j e^{f(x_i)_j}
        sum_i = 0.0
        for f_i_j in f_i:
            sum_i += np.exp(f_i_j)
        loss += -f_i[y[i]] + np.log(sum_i)

        # Compute gradient
        # dW_j = 1 / num_train * \ sum_i[x_i * (p(y_i = j) - Ind{y_i = j})]
        # Here we are computing the contribution to the inner sum for given i.
        for j in range(num_classes):
            p = np.exp(f_i[j]) / sum_i
            dW[j, :] += (p - (j == y[i])) * X[:, i]

    # Compute average
    loss /= num_train
    dW /= num_train

    # Regularization
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W
    
    return loss, dW
