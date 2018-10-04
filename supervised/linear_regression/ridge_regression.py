import numpy as np
from linear_regression import LinearRegression

class RidgeRegression(LinearRegression):

    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def fit(self, X, y, bias=True):
        '''
        Finds the parameters that minimises the 
        MSE + L2 norm of the data points to a straight line.

        Arguments:
                X: the features of the data points
                y: the labels of the data points
                bias: specify whether a bias should be used or not

        Returns:
                theta: the found parameters
        '''
        if bias:
            bias = np.ones((X.shape[0], 1))
            X = np.hstack([X, bias])

        XT = X.T
        XTX = np.dot(XT, X)

        regularisation_term = self.lambd * np.identity(XTX.shape[0])
        inverse_XTX = np.linalg.inv(XTX + regularisation_term)

        inverse_XTX_XT = np.dot(inverse_XTX, XT)

        theta = np.dot(inverse_XTX_XT, y)
        self.theta = theta

        return theta

    def loss(self, y, y_hat):
        '''
        Calculates the L2 regularised loss between the predicted
        labels and the true labels.

        Arguments: 
                y: the true labels
                y_hat: the predicted labels

        Returns:
                loss: the loss between the real and predicted labels
        '''
        number_of_points = y.shape[0]

        residual = np.linalg.norm(y - y_hat)
        regularisation_term = self.lambd * np.dot(self.theta.T, self.theta)[0, 0]
        loss = 1. / number_of_points * residual + regularisation_term

        return loss
