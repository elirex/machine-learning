import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loop).
    Inputs have dimension D, there are C classes, and we operate on
    minibatches of N examples.

    Inputs:
    -W: A numpy array of shape (D, C) containing weights.
    -X: A numpy array of shape (N, D) containing a minibatch of data.
    -y: A numpy array of shape (N,) containing training labels; 
         y[i] = c means
    - reg: (float) regularization strength

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
    num_classes = W.shape[1]
    num_train = X.shape[0]

    for i in range(num_train):
        # Compute scores
        f_scores = X[i, :].dot(W)

        # logC
        f_max = np.max(f_scores)
        f_scores -= f_max

        # Compute loss: L_i = -f(x_i)_{y_i} + log(sum_j e^{f(x_i)_j})
        loss += -f_scores[y[i]] + np.log(np.sum(np.exp(f_scores)))

        # Compute gradient 
        for j in range(num_classes):
            # p = np.exp(f_scores[j]) / (np.sum(np.exp(f_scores)))
            # dW[:, j] += X[i, :] * (-1 * (j == y[i]))
            dW[:, j] += X[i, :] * (-1 * (j == y[i])) + np.exp(f_scores[j])/(np.sum(np.exp(f_scores)))

    # # Compute average
    loss /= num_train
    dW /= num_train

    # Regularization
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W
    
    return loss, dW



def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized verstion.

    Inputs and outputs are the same as softmax_loss_naive.
    """

    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    # Compute the softmax loss and its gradient using no explicit loops.
    
    # Step1: Remove the numeric instability

    # Get shapes
    num_classes = W.shape[1]
    num_train = X.shape[0]

    # Compute scores
    f = np.dot(X, W)

    # Normalization trick to avoid numerical instability.
    # Remove the max of each score column
    f_max = np.max(f).reshape(-1, 1)
    f -= f_max
    scores = np.exp(f)
    # f -= np.max(f)

    # Step2: Compute the loss
    # Summing everything across the # of samples
    scores_sums = np.sum(scores, axis=1)
    # Select all the valid scores
    scores_correct = scores[np.arange(num_train), y]
    f_correct = f[np.arange(num_train), y]
    loss = np.sum(-f_correct + np.log(scores_sums))

    # Step 3: Compute the gradient of the function
    sum = scores / (scores_sums.reshape(-1, 1))
    # later on, we're gonna need a binary matrix for adding the 1's inside of the dW[:, j]
    bi_matrix = np.zeros_like(scores)
    bi_matrix[np.arange(num_train), y] = -1

    # The, recall we need to either add 1 or subtract 1 to each element if it's in the correct class
    sum += bi_matrix

    # Then, we will multiply it elementwise by X_i(this is kind of weird) to get a 3D array of NxDxC
    dW = (X.T).dot(sum)

    # Regularization
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    dW /= num_train
    dW += reg * W
    
    return loss, dW
