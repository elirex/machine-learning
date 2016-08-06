import numpy as np
from random import randrange


def eval_numerical_gradient(f, x):
    """
    A naive implementation of numerical gradient of f at x
    - f should be a function that takes a single argument
    - x is the point (numpy array) to evaluate the gradient at
    """
    
    fx = f(x) # Evaluate function value at original point
    grad = np.zeros(x.shape)
    h = 0.00001

    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        # Evaluate function at x + h
        ix = it.multi_index
        x[ix] += h # Increment by h
        fxh = f(x) # Evalute f(x + h)
        x[ix] -= h # Restore to previous value (very important!)

        # Compute the partial derivate
        grad[ix] = (fxh - fx) / h # The slope
        print(ix, grad[ix])
        it.iternext() # Step to next dimension

    return grad


def grad_check_sparse(f, x, analytic_grad, num_checks):
    """
    Sample a few random elements and only return numerical in the dimensions.
    """
    h = 1e-5

    for i in range(num_checks):
        ix = tuple([randrange(m) for m in x.shape])

        x[ix] += h # Increment by h
        fxph = f(x) # Evaluate f(x + h)
        x[ix] -= 2 * h # Increment by h
        fxmh = f(x) # Evaluate f(x - h)
        x[ix] += h # Reset

        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = analytic_grad[ix]
        rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
        print('Numerical: {0:f}, Analytic: {1:f}, Relative error: {2:e}'.format(
            grad_numerical, grad_analytic, rel_error))

