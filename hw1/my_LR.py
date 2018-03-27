"""
線性迴歸實作
python 3.5
Author:daniel-code
Date:2018.03.20
"""

import numpy as np


class my_LR():
    def __init__(self, alpha=0.1, tolerance=0.02, max_iterations=1000):
        """
        initial function
        :param alpha: learning rate
        :param tolerance: min cost to break
        :param max_iterations: max iteration to stop
        """
        # alpha is the learning rate or size of step to take in
        # the gradient decent
        self._alpha = alpha
        self._tolerance = tolerance
        self._max_iterations = max_iterations
        # thetas is the array coeffcients for each term
        # the y-intercept is the last element
        self._thetas = None

    def fit(self, X, y):
        """
        fit linear model
        :param X: feature data
        :param y: label data
        :return: coefficient of model
        """
        num_examples, num_features = np.shape(X)
        self._thetas = np.ones(num_features)

        xs_transposed = X.transpose()
        for i in range(self._max_iterations):
            # difference between our hypothesis and actual values
            diffs = np.dot(X, self._thetas) - y
            # print('diffs = ',diffs)
            # sum of the squares
            cost = np.sum(diffs ** 2) / (2 * num_examples)
            # print('cost = ', cost)
            # calculate averge gradient for every example
            gradient = np.dot(xs_transposed, diffs) / num_examples
            # update the coeffcients
            self._thetas = self._thetas - self._alpha * gradient
            # print(self._thetas)
            # check if fit is "good enough"
            if cost < self._tolerance:
                return self._thetas
        return self._thetas

    def get_Params(self):
        return {'alpha': self._alpha,
                'tolerance': self._tolerance,
                'max_iterations': self._max_iterations}

    def predict(self, X):
        """
        predict for intput data x
        :param X: intput feature
        :return: predect results
        """
        return np.dot(X, self._thetas)

    def score(self, X, y):
        """
        measure MSE for model by X data
        :param X: test intput feature
        :param y: test label data
        :return: mse
        """
        predict = self.predict(X)
        diffs = predict - y
        y_true_mean = np.array(y).mean()

        return 1 - np.sum(diffs ** 2) / np.sum((y - y_true_mean) ** 2)
