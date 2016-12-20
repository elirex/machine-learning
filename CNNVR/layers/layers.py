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

def batchnorm_backward(dout, cache):
    """
        Backward pass for batch normalization.

        For this implementation, you should write out a computation graph for
        batch normalization on paper and propagate gradients backward through
        intermediate nodes.

        Inputs:
        - dout: Upstream derivatives, of shape (N, D)
        - cache: Variable of intermediates from batchnorm_forward.

        Returns a tuple of:
        - dx: Gradient with respect to inputs x, of shape (N, D)
        - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
        - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    
    dx, dgamma, dbeta = None, None, None
    # Unwrap all this stuff
    u_b = cache['mean']
    sigma_squared_b = cache['variance']
    x_hat = cache['x_hat']
    gamma = cache['gamma']
    beta = cache['beta']
    eps = cache['eps']
    x = cache['x']
    N = x.shape[0]

    # Compute derivatives with respect to x (notation is x_1 if it's the 
    # first backwards)
    dx_1 = gamma * dout

    # When we multiply a. (x-u_b) by b. (sigma_squared_b + eps)^-0.5
    dx_2_b = np.sum((x-u_b) * dx_1, axis=0)
    dx_2_a = ((sigma_squared_b + eps) ** 0.5) * dx_1

    # When we have (sigma_squared_b + eps) ^ -0.5
    dx_3_b = (-0.5) * ((sigma_squared_b) ** -1.5) * dx_2_b

    # When we have addition of epsilon
    dx_4_b = dx_3_b * 1

    # When we have the summation of calculating sigma
    dx_5_b = np.ones_like(x) / N * dx_4_b

    # When we have to the power of 2 of calculating sigma
    dx_6_b = 2 * (x-u_b) * dx_5_b

    # Whe we have to congregate both sources of dout1 and dout2
    # In addition, we're also adding, so just multiply by 1 to show that
    dx_7_a = dx_6_b * 1 + dx_2_a * 1
    dx_7_b = dx_6_b * 1 + dx_2_a * 1

    # When multiplied by -1 (so we can magate the adding to a subtract),
    # value is -1 * prev_val
    dx_8_b = -1 * np.sum(dx_7_b, axis=0)

    # Whe we have summation of calculating mean
    dx_9_b = np.ones_like(x) / N * dx_8_b

    # Whe we have to congregate both sources of dout1 and dout2
    dx_10 = dx_9_b + dx_7_a

    dgamma = np.sum(x_hat * dout, axis=0)
    dbeta = np.sum(dout, axis=0)
    dx = dx_10

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter, We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but
        not in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: A tuple (dropout_param, mask). In training mode, mask is the 
        dropout mask that was used to multiply the input; in test mode, mask 
        is None.
    """
    
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None
    if mode == 'train':
        mask = (np.random.rand(*x.shape) > p) / p
        out = x * mask
    elif mode == 'test':
        out = x

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) droput.

    Inputs:
    - dout: Upstream derivatives, of any shape.
    - cache: (dropout_param, mask) from dropout_forward.
    """

    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        dx = dout * mask
    elif mode == 'test':
        dx = dout

    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
        A naive implementation of the forward pass for a convolutional layer.

        The input consists of N data points, each with C channels, height H and
        width W. We convolve each input with F different filters, where each
        filter spans all C channels and height HH and width WW

        Input:
        - x: Input data of shape (N, C, H, W)
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)
        - conv_param: A dictionary with the following keys:
            - 'stride': The number of pixels between adjacent receptive fields
                in the horizontal and vertical directions
            - 'pad': The number of pixels that will be used to zero-pad the input.

        Returns a tuple of:
        - out: Output data, of shape (N, F, H', W') where H' and W' are given by
            H' = 1 + (H + 2 * pad - HH) / stride
            W' = 1 + (W + 2 * pad - WW) / stride
        - cache: (x, w, b, conv_param)
    """

    stride = conv_param['stride']
    pad = conv_param['pad']
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    out = None

    # Padding
    # x.shape = (1, 1, 3, 3)
    # x = [[[[1, 1, 1],
    #        [1, 1, 1],
    #        [1, 1, 1],]]]
    # padding 1
    # x = [[[[0, 0, 0, 0, 0],
    #        [0, 1, 1, 1, 0],
    #        [0, 1, 1, 1, 0],
    #        [0, 1, 1, 1, 0],
    #        [0, 0, 0, 0, 0]]]]
    x_pad = np.pad(x, pad_width=[(0,), (0,), (pad,), (pad,)], mode='constant',
            constant_values=0)
    H_prime = (H + 2 * pad - HH) / stride + 1
    W_prime = (W + 2 * pad - WW) / stride + 1
    out = np.zeros((N, F, H_prime, W_prime))
    for i in xrange(H_prime):
        for j in xrange(W_prime):
            selected_x = x_pad[:, :, i * stride : i * stride + HH, 
                                j * stride : j * stride + WW]
            for k in xrange(F):
                out[:, k, i, j] = np.sum(selected_x * w[k], axis(1, 2, 3)) + b[k]
    
    cache = (x, w, b, conv_param)
    return out, cache

def conv_backward_naive(dout, cache):
    """
        A naive implementation of the backward pass for a convolutional layer.

        Inputs:
        - dout: Upstream derivatives.
        - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

        Returns a tuple of:
        - dx: Gradient with respect to x
        - dw: Gradient with respect to w
        - db: Gradient with respect to b
    """
    
    dx, dw, db = None, None, None
    # Unwrap cache
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']

    x_pad = np.pad(x, pad_width=[(0,), (0,), (pad,), (pad,)], mode='constant',
            constant_values=0)

    # Shape the numpy arrays
    dx_pad = np.zeros_like(x_pad) # We will trim off the dx's on the paddings later
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    # db calculation
    db = np.sum(dout, axis=(0, 2, 3))

    # Loading up some values before going to calculate dw and dx
    H_prime = (H + 2 * pad - HH) / stride + 1
    W_prime = (W + 2 * pad - WW) / stride + 1

    for i in xrange(H_prime):
        for j in xrange(W_prime):
            selected_x = x_pad[:, :, i*stride: i*stride+HH,
                    j * stride: j * stride + WW]
            selected_shape = selected_x.shape
            for k in xrange(F):
                dw[k] += np.sum(selected_x * (dout[:, k, i ,j])[:, None, None, None], axis=0)
            dx_pad[:, :, i*stride: i*stride+HH, j*stride: j*stride+WW] +=\
                    np.einsum('ij,jklm->iklm', dout[:, :, i, j], w)
    dx = dx_pad[:, :, pad:-pad, pad:-pad]

    return dx, dw, db
