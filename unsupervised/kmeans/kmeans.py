import numpy as np


class KMeans(object):
    '''
    A class representing the KMeans, containing
    all the method necessary to perform KMeans.
    '''

    def __init__(self, number_of_clusters):
        '''
        Constructor defining the number of
        clusters to use for KMeans.
        '''
        self.number_of_clusters = number_of_clusters

    def initialise_centroids(self, X):
        '''
        Initialises the position of the centroids
        based on a few data points in the data.

        Arguments:
            X: the feature data.

        Returns:
            centroids: the instantiated centroids.
        '''

        number_of_data = X.shape[0]
        indices = np.random.randint(
            number_of_data, size=self.number_of_clusters)

        centroids = X[indices]

        return centroids

    def euclidean_distance(self, point1, point2):
        '''
        Calculates the euclidean distance between
        two points.

        Arguments:
            point1: the first data point.
            point2: the second data point.

        Returns:
            distance: the euclidean distance between
                      point1 and point2
        '''

        distance = np.linalg.norm(point1 - point2)

        return distance

    def closest_centroid(self, data_point, centroids):
        '''
        Finds the closest centroid to a single data point.
        Arguments:
            data_point: the data point under consideration.
            centroids: the position of all the centroids.

        Returns:
            closest: the index of the closest centroid
                     to data_point
        '''

        distance = [self.euclidean_distance(
            data_point, centroid) for centroid in centroids]
        closest = np.argmin(distance)

        return closest

    def centroid_mean(self, X, assignment):
        '''
        Given the data-set and their centroid assignments,
        this method finds the position of the centroids
        for each cluster.

        Arguments:
            X: the feature data.
            assignment: the centroid assignments of
                        each data point

        Returns:
            centroids: the position of the centroids
        '''

        centroids = []
        for i in range(self.number_of_clusters):
            cluster = [index for index, cluster in enumerate(
                assignment) if i == cluster]

            data_in_cluster = X[cluster]

            centroid_avg = np.mean(data_in_cluster, axis=0)
            centroids.append(centroid_avg)

        centroids = np.array(centroids)

        return centroids

    def fit(self, X):
        '''
        Run the KMeans algorithm for some
        feature data and assigns each data
        point to a cluster.

        Arguments:
            X: the feature data.

        Returns:
            assignment: the assignments of each
                        data point to the clusters.
        '''
        centroids = self.initialise_centroids(X)

        cache_centroids = np.zeros(centroids.shape)

        while self.euclidean_distance(centroids, cache_centroids) > 0.001:
            cache_centroids = centroids

            assignment = list(
                map(lambda data_point: self.closest_centroid(data_point, centroids), X)
                )

            centroids = self.centroid_mean(X, assignment)

        return assignment
