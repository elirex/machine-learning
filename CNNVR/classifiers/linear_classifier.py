import numpy as np

class LinearClassifier:

    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
        """
        Train this linear classifier using stochastic gradient descent.

        Inputs:
        - X: D x N array of training data. Each training point is a D-dimensional
        - y: 1-dimensional array of length N with labels 0...K-1, for K classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing.
        - batch_size: (integer) number of traiing examples to use at each step.
        - verbose: (boolean) If ture, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """

        dim, num_train = X.shape # Example X.shape(3073, 49000)
        num_classes = np.max(y) + 1 # Assume y takes values 0,,,K-1 where K is number of classes
        if self.W is None:
            # lazily initialize W
            self.W = np.random.randn(num_classes, dim) * 0.01

        # Run stochatic gradient descent to optimize W
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None

            # Sample batch_size element from the training data and their
            # corresponding labels to use in this round of gradient descent.
            indices = np.array([np.random.choise(num_train) for i in range(batch_size)])
            X_batch = X[:, indices]
            y_batch = y[indices]

            # Evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # TODO: Update the weights using the gradient and the learning rate.
            self.W += -learning_rate * grad

            if varbose and it % 100 == 0:
                print('Iteration {0:d} / {1:d} loss {2:f}'.format(it, num_iters, loss))
        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels 
        for data points.

        Inputs:
        - X: D x N array of training data. Each column is a D-dimensional 
                point.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 
                1-dimensional array of length N, and each element is an 
                integer giving the predicted class.
        """
        y_pred = np.zeros(X.shape[1])
        
        # TODO: Implement this method. Store the predicted labels in y_pred.

        score = self.W.dot(X)
        y_pred = np.argmax(score, axis=0)

        return y_pred

    def loss(self, X_batch, y_batch, reg):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - X_batch: D x N array of data; each column is a data point.
        - y_batch: 1-dimensional array of length N with labels 0...K-1, for  K classes.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float.
        - gradient with recpect to self.W; an array of the same shape as W
        """
        pass



class LinearSVM(LinearClassifier):
    """ A subclass that uses the Multiclass SVM loss function """

