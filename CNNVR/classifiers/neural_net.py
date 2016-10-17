import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network.

    The network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each
    class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """

        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)


    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two-layer fully connected 
        neural network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each
            y[i] is an integer in the range 0 <= y[i] < C. This parameter is
            optional; if it is not passed then we only return scores, and if
            it is passed then we instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where 
        scores[i, c] is the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        -loss: Loass (data loss and regularization loss) for thie batch of
            training samples.
        - grads: Dictionary mapping parameter names to gradients of those 
            parameters with respect to the loss function; has the same keys
            as self.params.
        """

        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        scores = None
        ###################################################################
        # TODO: Perform the forward pass, computing the class scores for  #
        # input. Store the result in the scores variable, with should be  #
        # an array of shape (N, C).                                       #
        ###################################################################
        # Fully connected Layer (ReLU)
        hidden = np.dot(X, W1) + b1
        h_scores = np.maximum(0, hidden)
        scores = np.dot(h_scores, W2) + b2
        ###################################################################
        #                        END OF YOUR CODE                         #
        ###################################################################

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        ###################################################################
        # TODO: Finish the forward pass, and compute the loss. This       #
        # should include both the data loss and L2 regularization for W1  #
        # and W2. Store the result in the variable loss, which should be  #
        # a scalar. Use the softmax classifier loss. So that your results # 
        # match ours ,multiply the regularization loss by 0.5.            #
        ###################################################################
       
        # Pre-compute the max of the exponentiation and subtract the value
        # by it.
        scores_max = np.max(scores).reshape(-1, 1)
        scores -= scores_max
        scores_exp = np.exp(scores)
        scores_exp_sum = np.sum(scores_exp, axis=1)
        correct_scores = scores[np.arange(N), y] 
        loss = np.sum(-correct_scores + np.log(scores_exp_sum)) / N

        # The regularization:
        loss += 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        ###################################################################
        #                        END OF YOUR CODE                         #
        ###################################################################

        # Backward pass: compute gradients
        grads = {}
        ###################################################################
        # TODO: Compute the backward pass, computing the derivatives of   #
        # the weights and biases. Store the results in the grads          #
        # dictionary. For example, grads['W1'] should store the gradient  #
        # on W1, and be a matrix of same size.                            #
        ###################################################################

        ###################################################################
        #                        END OF YOUR CODE                         #
        ###################################################################

        return loass, grads
