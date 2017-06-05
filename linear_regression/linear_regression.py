import numpy as np
import matplotlib.pyplot as plt
from operator import add

class LinearRegression:
	def __init__(self, x_data, y_data, degree):
		self.x_data = self.instantiate_data(x_data, degree)
		self.y_data = np.matrix(y_data).T

		self.params = np.ones((1, self.x_data[0].size)).T
		self.residuals = []

	@property
	def number_of_points(self):
		return len(self.y_data)

	def instantiate_data(self, x_data, degree):
		x_matrix = np.ones(len(x_data))

		for i in range(1, degree + 1):
			x_matrix = np.c_[x_matrix, [x**i for x in x_data]]

		return x_matrix

	def model(self, x, params):
		return np.dot(x, params)

	def cost(self, x, y, params):
		return 1./(2 * len(x)) * np.sum(np.square(self.model(x, params) - y))

	def gradient_descent(self, learning_rate, regularisation, num_iterations=10000):
		temp_params = []
		iters = 0

		while iters < num_iterations:
			new_param = self.params[0] - learning_rate / self.number_of_points * np.sum(np.dot(np.matrix(self.x_data[:, 0]), self.model(self.x_data, self.params) - self.y_data))
			temp_params.append(new_param)

			for index, param in enumerate(self.params[1:]):
				new_param = param * (1 - learning_rate * regularisation / (self.number_of_points)) - learning_rate / self.number_of_points * np.sum(np.dot(np.matrix(self.x_data[:, index + 1]), self.model(self.x_data, self.params) - self.y_data))
				temp_params.append(new_param)
			
			self.params = np.array(temp_params)

			iters += 1
			temp_params = []

			training_cost = self.cost(self.x_data, self.y_data, self.params)
			if iters % 100 == 0:
				print("After " + str(iters) + " iterations: " + str(training_cost))

			self.residuals.append(training_cost)

		return self.params, training_cost	

	def evaluate(self, X_test, y_test):
		return self.cost(X_test, y_test, self.params)
	
