import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in range(num_train):
      scores = X[i].dot(W) # 1 by C
      scores_max = np.max(scores)
      scores -= scores_max
      scores_exp = np.exp(scores)
      scores_exp_sum = np.sum(scores_exp) 
      prob = scores_exp[y[i]] / scores_exp_sum
      loss += -np.log(prob)
      for j in range(num_classes):
          prob_j = scores_exp[j] / scores_exp_sum
          if j == y[i]:
              dW[:, j] += (prob_j - 1) * X[i]
          else:
              dW[:, j] += prob_j * X[i]
      # Computes the loss
      # scores_sum = np.sum(scores_exp)
      # loss += -scores[y[i]] + np.log(scores_sum)
      # for j in range(num_classes):
      #     dW[:, j] += X[i] * (-1*(j==y[i]) + np.exp(scores[j])/scores_sum)

  loss /= num_train
  dW /= num_train
  loss += reg * np.sum(W*W) * 0.5
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  scores = X.dot(W) # N by C
  max_scores = np.max(scores)
  scores -= max_scores
  exp_scores = np.exp(scores)

  sum_scores = np.sum(exp_scores, axis=1, keepdims=True)

  # print('sum_scores: ', sum_scores)
  probs = exp_scores / sum_scores
  corect_log_probs = -np.log(probs[np.arange(num_train), y])
  loss = np.sum(corect_log_probs) / num_train
  loss += 0.5 * reg * np.sum(W * W)

  dscores = probs
  dscores[np.arange(num_train), y] -= 1
  dscores /= num_train
  dW = X.T.dot(dscores)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

