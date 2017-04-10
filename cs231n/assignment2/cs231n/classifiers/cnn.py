import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################

    
    # Conv layer
    # The parameters of the conv is of size (F, C, HH, WW) with F give the 
    # nb of filters, C, HH, WW characterize the size of each filter.
    # Input size: (N, C, H, W)
    # Output size: (N, F, Hc, Wc)

    # Size of the input
    C, H, W = input_dim

    F = num_filters
    filter_height = filter_size
    filter_width = filter_size
    stride_conv = 1 # Stride
    P = (filter_size - 1) / 2 # padd
    Hc = (H + 2 * P - filter_height) / stride_conv + 1
    Wc = (W + 2 * P - filter_width)  / stride_conv + 1

    W1 = weight_scale * np.random.randn(F, C, filter_height, filter_width)
    b1 = np.zeros(F)

    # Pool layer: 2 x 2
    # The pool layer has no parameters but is important in the count of dimension.
    # Input size: (N, F, Hc, Wc)
    # Output size: (N, F, Hp, Wp)
    
    height_pool = 2
    width_pool = 2
    stride_pool = 2
    Hp = (Hc - height_pool) / stride_pool + 1
    Wp = (Wc - width_pool) / stride_pool + 1

    # Hidden Affine layer
    # Size of the parameter (F * Hp * Wp, H1)
    # Input size: (N, F * Hp * Wp)
    # Output size: (N, Hh)

    Hh = hidden_dim
    W2 = weight_scale * np.random.randn(F * Hp * Wp, Hh)
    b2 = np.zeros(Hh)

    # Output affine layer
    # Size of the parameter (Hh, Hc)
    # Input size: (H, Hh)
    # Output size: (N, Hc)

    Hc = num_classes
    W3 = weight_scale * np.random.randn(Hh, Hc)
    b3 = np.zeros(Hc)

    self.params.update({'W1': W1,
                        'W2': W2,
                        'W3': W3,
                        'b1': b1,
                        'b2': b2,
                        'b3': b3})

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'


    N = X.shape[0]


    # Forward into the conv layer
    x = X
    w = W1
    b = b1
    conv_layer, cache_conv_layer = conv_relu_pool_forward(
            x, w, b, conv_param, pool_param)
    N, F, Hp, Wp = conv_layer.shape # Output shape

    # Forward into the hidden layer
    x = conv_layer.reshape((N, F * Hp * Wp))
    w = W2
    b = b2
    
    hidden_layer, cache_hidden_layer = affine_relu_forward(x, w, b)
    N, Hh = hidden_layer.shape
    
    # Forward into the linear output layer
    x = hidden_layer
    w = W3
    b = b3
    scores, cache_scores = affine_forward(x , w, b)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dout = softmax_loss(scores, y)

    reg_loss = 0.5 * self.reg * np.sum(W1**2)
    reg_loss += 0.5 * self.reg * np.sum(W2**2)
    reg_loss += 0.5 * self.reg * np.sum(W3**2)
    loss = loss + reg_loss

    # Backpropagation
    # Backprop inot output layer
    dx_3, dw_3, db_3 = affine_backward(dout, cache_scores)
    dw_3 += self.reg * W3

    # Backprop inot first layer
    dx_2, dw_2, db_2 = affine_relu_backward(dx_3, cache_hidden_layer)
    dw_2 += self.reg * W2

    # Backprop into the conv layer
    dx_2 = dx_2.reshape(N, F, Hp, Wp)
    dx_1, dw_1, db_1 = conv_relu_pool_backward(dx_2, cache_conv_layer)

    dw_1 += self.reg * W1

    grads.update({'W1': dw_1,
                  'W2': dw_2,
                  'W3': dw_3,
                  'b1': db_1,
                  'b2': db_2,
                  'b3': db_3})


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
