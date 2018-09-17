import numpy as np
import matplotlib.pyplot as plt

def sign(x):
	if(x >= 0):
		return 1
	else:
		return -1

class Perceptron(object):
	def __init__(self, num_inputs):
		self.weights = -1  + 2 * np.random.rand(num_inputs + 1)
		self.errors = []

	def evaluate(self, data_point):
		data_vector = np.append(1, data_point.get_points())
		return sign(self.weights.dot(data_vector))

	def func_evaluate(self, x):
		return -self.weights[0] / self.weights[2] - self.weights[1] * x / self.weights[2]
		 
	def error(self, data_point):
		predicted_label = self.evaluate(data_point)
		error =  data_point.label - predicted_label
		return error

	def train(self, data_point, learning_rate):
		error = self.error(data_point)
		data_point_bias = np.append(1, data_point.get_points())

		for i in range(len(self.weights)):
			self.weights[i] += error * data_point_bias[i] * learning_rate
	
	def store_errors(self, data_points):
		num_error = len(filter(lambda x: x != 0, map(lambda point: self.error(point), data_points)))
		self.errors.append(num_error)
			
		
class Point(object):
	def __init__(self, x, y, func=None):
		self.x = x
		self.y = y

		if func != None:
			if y > func(x):
				self.label = 1
			else:
				self.label = -1

	def get_points(self):
		return [self.x, self.y]
