import numpy as np
import matplotlib.pyplot as plt


class Kernel(object):
	'''
	A class that stores different types of kernels.
	'''

    def polynomial(x, z, n=2):
        kernel_matrix = (1 + np.dot(x, z))**n

        return kernel_matrix

    def rbf(x, z, sigma=1):
    	kernel_matrix = np.zeros((x.shape[0], z.shape[0]))

    	for i in range(len(x)):
    		for j in range(len(z)):
    			kernel_matrix[i, j] = np.exp(-np.linalg.norm(x[i, :] - z[j, :]) / (2 * sigma**2))

    	return kernel_matrix


class KernelRidgeRegression(object):
	'''
	A class that represents a kernel ridge regression
	model.
	'''

    def __init__(self, kernel, lambd):
        self.lambd = lambd
        self.kernel = kernel

    def transform(self, x, z):
        return self.kernel(x, z)

    def fit(self, X, y, bias=True):
        '''
        Finds the parameters that minimises the 
        MSE of the data points to a straight line.

        Arguments:
                X: the features of the data points
                y: the labels of the data points
                bias: specify whether a bias szould be used or not

        Returns:
                theta: the found parameters
        '''
        K = self.transform(X, X)
        lambdaI = self.lambd * np.identity(K.shape[0])

        alpha = np.dot(np.linalg.inv(K + lambdaI), y)

        self.X = X
        self.alpha = alpha

        return alpha

    def predict(self, X, bias=True):
        '''
        Given data points, predicts what the
        labels will be.

        Arguments: 
                X: the features of the data points
                bias: specify whether a bias should be used or not.
        '''
        y_hat = np.dot(self.transform(X, self.X), self.alpha)

        return y_hat
