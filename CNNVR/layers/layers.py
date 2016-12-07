import numpy as np

def affine_forward(x, w, b):
    """
        Computes the forward pass for an affine (fully-connected) layer.

        The input x has shape(N, d_1, ..., d_k) and contains a minbatch of N
        examples, where each example x[i] has shape (d_1, ..., d_k). We will
        reshape each input into a vector of dimension D = d_1 * ... * d_k, and
        then transform it to an output vector of dimeansion M.

        Inputs:
        - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
        - w: A numpy array of weights, of shape (D, M)
        - b: A numpy array of biases, of shape (M,)

        Returns a tuple of:
        - out: output, of shape (N, M)
        - cache: (x, w, b)
    """

    out = None
    # Unroll the vector
    N = x.shape[0]
    x_row = x.reshape(N, -1)
    out = x_row.dot(w) + b
    cache = (x, w, b)
    return out ,cache


def affine_backward(dout, cache):
    """
        Computes the backward pass for an affine layer.

        Inputs:
        - dout: Upstream derivative, of shape (N, M)
        - cache: Tuple of:
            - x: Input data, of shape (N, d_1, ..., d_k)
            - w: Weights, of shape (D, M)
            - b: Biases, of shape (M,)

        Returns a tuple of:
        - dx: Gradient with respect to x, of shape (N, d_1, ..., d_k)
        - dw: Gradient with respect to w, of shape (D, M)
        - db: Gradient with respect to b, of shape (M,)
    """

    x, w, b = cache
    dx, dw, db = None, None, None
    # Unroll the vector
    x_shape = x.shape
    N = x.shape[0]
    x_row = x.reshape(N, -1)

    dx = (dout.dot(w.T)).reshape(x_shape)
    db = np.sum(dout, axis=0)
    dw = (x_row.T).dot(dout)

    return dx, dw, db


def relu_forward(x):
    """
        Computes the forward pass for a layer of rectified linear units (ReLUs)

        Inputs:
        - x: Inputs, of any shape

        Returns a tuple of:
        - out: Output, of the same shape as x
        - cache: x
    """

    out = None

    binary_matrix = np.ones(x.shape)
    binary_matrix[x < 0] = 0
    out = x * binary_matrix
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
        Computes the backward pass for a layer of rectified linear units (ReLUs)

        Inputs:
        - dout: Upstream derivatives, of any shape
        - cache: Input x, of same shape as dout

        Returns:
        = dx: Gradient with respect to x
    """

    dx, x = None, cache
    dx = np.once(x.shape) * dout
    dx[x < 0] = 0

    return dx


def batchnorm_forward(x, gamma, bate, bn_param):
    """
        Forward pass for batch normalization.

        During training the sample mean and (uncorrected) sample variance are
        computed from minibatch statistics and used to normalize the incoming
        data. During training we also keep an exponentially decaying running
        mean of the mean and variance of each feature, and these averages are
        used to normalize data at test-time.

        At each timestep we update the running averages for mean and variance 
        using an exponential decay based on the momentum parameter:

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + 1 (1 - momentum) * sample_var

        Note that the batch normalization paper suggests a different test-time
        behavior: they compute sample mean and variance for each feature using
        a large number of training images rather than using a running average.
        For this implementation we have chosen to use running averages instead
        since they do not require an additional estimation ste; the torch7 
        implementation of batch normalization also uses running avarages.

        Input:
        - x: Data of shape(N, D)
        - gamma: Scale parameter of shape (D,)
        - beta: Shift parameter of shape (D,)
        - bn_param: Dictionary with the following keys:
            - mode: 'train' or 'test'; required
            - eps: Constant for numeric stability
            - mementum: Constant for running mean / variance.
            - running_mean: Array of shape (D,) giving running mean of features
            - running_var: Array of shape (D,) giving running variance of features

        Returns a tuple of:
        - out: of shape (N, D)
        - cache: A tuple of values needed in the backward pass
    """

    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('mementum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))
    
    out, cahce = None, {}
    if mode == 'train':
        # Get the mean of every feature
        u_b = (np.sum(x, axis=0)) / N

        # Get teh variance
        sigma_squared_b = np.sum((x-u_b)**2, axis=0) / N

        # Git x_hat
        x_hat = (x-u_b) / np.sqrt(sigma_squared_b + eps)
        out gamma * x_hat + beta

        cache['mean'] = u_b
        cache['variance'] = sigma_squared_b
        cache['x_hat'] = x_hat
        cache['gamma'] = gamma
        cahce['beta'] = beta
        cache['eps'] = eps
        cache['x'] = x

        # Keeping track of running mean and var
        running_mean = momentum * running_mean + (1 - momentum) * u_b
        running_var = momentum * running_var + (1 - momentum) * sigma_squared_b
    elif mode == 'test':
        x_hat = (x - running_mean.reshape(1, -1)) / 
            np.sqrt(running_var + eps).reshape(1, -1)
        out = gamma * x_hat + beta

        cache['x_hat'] = x_hat
        cache['gamma'] = gamma
        cahce['beta'] = beta
        cache['eps'] = eps
        cache['x'] = x
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means and var back into np_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var
    return out, cache
