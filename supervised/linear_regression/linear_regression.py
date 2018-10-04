import numpy as np

class LinearRegression(object):
    '''
    A class that represents a Linear Regression classifier
    '''

    def fit(self, X, y, bias=True):
        '''
        Finds the parameters that minimises the 
        MSE of the data points to a straight line.

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

        inverse_XTX = np.linalg.inv(XTX)

        inverse_XTX_XT = np.dot(inverse_XTX, XT)

        theta = np.dot(inverse_XTX_XT, y)
        self.theta = theta

        return theta

    def predict(self, X, bias=True):
        '''
        Given data points, predicts what the
        labels will be.

        Arguments: 
                X: the features of the data points
                bias: specify whether a bias should be used or not.
        '''
        if bias:
            bias = np.ones((X.shape[0], 1))
            X = np.hstack([X, bias])

        y_hat = np.dot(X, self.theta)

        return y_hat

    def loss(self, y, y_hat):
        '''
        Calculates the loss between the predicted
        labels and the true labels.

        Arguments: 
                y: the true labels
                y_hat: the predicted labels

        Returns:
                loss: the loss between the real and predicted labels
        '''
        number_of_points = y.shape[0]
        loss = 1. / number_of_points * np.linalg.norm(y - y_hat)

        return loss
