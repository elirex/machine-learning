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
  num_class = W.shape[1]
  # print('X.shape:', X.shape)
  # print('W.shape:', W.shape)
  loss = 0.0
  for i in range(num_train):
        score_i = X[i].dot(W)
        # print('score_i.shape:', score_i.shape)
        stability = -np.max(score_i)
        # print('stability:', stability)
        # print('stability.shape:', stability.shape)
        exp_score_i = np.exp(score_i + stability)
        exp_score_sum_i = np.sum(exp_score_i)
        for j in range(num_class):
            if j == y[i]:
                dW[:, j] += -X[i] + (exp_score_i[j] / exp_score_sum_i) * X[i]
            else:
                dW[:, j] += (exp_score_i[j] / exp_score_sum_i) * X[i]
        numerator = np.exp(score_i[y[i]] + stability)
        # denom = np.sum(np.exp(score_i + stability))
        loss += -np.log(numerator / exp_score_sum_i)
   
  loss = loss / float(num_train) + 0.5 * reg * np.sum(W*W)
  dW = dW / float(num_train) + reg * W
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
  num_class = W.shape[1]
  
  scores = np.dot(X, W) # (N, D) x (D, C) = (N, C)
  # print('scores.shape:', scores.shape)
  scores -= np.max(scores)
  # print('(scores - max_scores).shape:', scores.shape)
  exp_scores = np.exp(scores)
  # print('exp_scores.shape:', exp_scores.shape)

  sum_exp_scores = np.sum(exp_scores, axis=1, keepdims=True)
  # print('sum_exp_scores.shape:', sum_exp_scores.shape)
  probs = exp_scores / sum_exp_scores
  # print('probs.shape:', probs.shape)
  correct_logprobs = -np.log(probs[np.arange(num_train), y])
  # print('correct_logprobs.shape:', correct_logprobs.shape)
  data_loss = np.sum(correct_logprobs) / num_train
  reg_loss = 0.5 * reg * np.sum(W*W)
  loss  = data_loss + reg_loss
    
  dscores = probs
  dscores[np.arange(num_train), y] -= 1
  dscores /= num_train
  # print('dscores.shape:', dscores.shape)
  dW = np.dot(X.T, dscores) # (N, D).T x (N, C) = (D, C)
  dW += reg * W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

