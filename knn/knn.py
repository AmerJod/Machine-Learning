import numpy as np
import matplotlib.pyplot as plt

class Neighbor(object):
	def __init__(self, test_point, train_point, distance):
		self.test_point = test_point
		self.train_point = train_point
		self.distance = distance

class KNearestNeighbors(object):
	def __init__(self, k):
		self.k = k

	def accuracy(self, testing_data, classified_data):
		mistakes = 0

		for test_data, classy_data in zip(testing_data, classified_data):
			print(test_data.label, classy_data.label)
			if(test_data.label != classy_data.label):
				mistakes += 1

		return float((len(testing_data) - mistakes))/len(testing_data)

	def classify_data(self, training_data, testing_data):
		classified_points  = []

		for test_point in testing_data:
			neighbours = []
			for train_point in training_data:
				neighbours.append(Neighbor(test_point, train_point, self.euclidean_distance(test_point, train_point)))

			neighbours = sorted(neighbours, key=lambda neighbour: neighbour.distance)[:self.k]

			label_count = self.count_neighbour_labels(neighbours)

			if(label_count[0] > label_count[1]):
				point = Point(test_point.x, test_point.y)
				point.label = 1
			else:
				point = Point(test_point.x, test_point.y)
				point.label = -1

			classified_points.append(point)

		return classified_points

	def count_neighbour_labels(self, neighbours):
		label_count = [0, 0]

		for neighbour in neighbours:
			if neighbour.train_point.label == 1:
				label_count[0] += 1
			else:
				label_count[1] += 1

		return label_count


	def euclidean_distance(self, p1, p2):
		return np.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)


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

	def get_label(self):
		return self.label

#points = [Point(4 * np.pi * np.random.rand(), -1 + 2 * np.random.rand(), lambda x: np.cos(x)**2 - 0.5) for i in range(10000)]
points = [Point(-1 + 2 * np.random.rand(), np.random.rand(), lambda x: x**2) for i in range(10000)]
training_data = points[2000:]
testing_data = points[:2000]

knn = KNearestNeighbors(5)

classified_data = knn.classify_data(training_data, testing_data)

print(knn.accuracy(testing_data, classified_data))

for data in classified_data:
	if(data.label == 1):
		plt.scatter(data.x, data.y, label=data.label, color="g")
	else:
		plt.scatter(data.x, data.y, label=data.label, color="b")

plt.show()


