import numpy as np


class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.iterations = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # GD
        for _ in range(self.iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            residuals = y_pred - y

            # gradient of RSS for each weight w is dw = 2 * x * (y_pred - y)
            # for bias it is db = 2 * (y_pred - y)
            # we sum all the gradients from all samples and normalize it by the number of samples
            # => each epoch updates each weight only one time
            dW = (1 / n_samples) * 2 * np.dot(X.T, residuals)
            db = (1 / n_samples) * 2 * np.sum(residuals)

            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
