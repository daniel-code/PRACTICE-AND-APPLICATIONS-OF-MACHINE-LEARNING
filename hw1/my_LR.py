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

    def fit(self, xs, ys):
        """
        fit linear model
        :param xs: feature data
        :param ys: label data
        :return: coefficient of model
        """
        num_examples, num_features = np.shape(xs)
        self._thetas = np.ones(num_features)

        xs_transposed = xs.transpose()
        for i in range(self._max_iterations):
            # difference between our hypothesis and actual values
            diffs = np.dot(xs, self._thetas) - ys
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
        return np.sum(diffs ** 2) / len(X)
