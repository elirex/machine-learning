import numpy as np
##############################################################################
# K Nearest Neighbor Class
##############################################################################
class KNearestNeighbor(object):
    """ a KNN classifier with L2 distance """
    def __init__(self):
        pass

    def train(self, train_dataset, train_labels):
        self.train_dataset = train_dataset
        self.train_labels = train_labels

    def predict(self, dataset, k = 1, num_loops = 0):
        if num_loops == 0:
            dists = self.compute_distances_no_loop(dataset)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(dataset)
        elif num_loops == 2:
            dists = slef.compute_distances_two_loops(dataset)
        else:
            raise ValueError('Invalid value {0:d} for num_loops'.format(num_loops))
        
        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, dataset):
        num_test = dataset.shape[0]
        num_train = self.train_dataset.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                dists[i, j] = np.sqrt(np.sum(np.square(self.train_dataset[j, :] - dataset[i, :])))
        return dists

    def compute_distances_one_loop(self, dataset):
        num_test = dataset.shape[0]
        num_train = dataset.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dists[i, :] = np.sprt(np.sum(np.square(self.train_dataset - dtatset[i, :]), axis = 1))
        return dists

    def compute_distances_no_loop(self, dataset):
        num_test = dataset.shape[0]
        num_train = self.train_dataset.shape[0]
        dists = np.zeros((num_test, num_train))
        # (a-b)^2 = a^2 - 2ab - b^2
        test = np.square(dataset).sum(axis = 1)
        train = np.square(self.train_dataset).sum(axis = 1)
        M = np.dot(dataset, self.train_dataset.T)
        dists = np.sqrt(train + np.matrix(test).T -2 * M)
        return dists

    def predict_labels(self, dists, k = 1):
        num_test = dists.shape[0]
        predict = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbor to
            # the ith test point.
            closet_labels = []
            indexs = np.argsort(dists[i, :])
            for i in range(0, k):
               closest_labels.append(self.train_dataset[index[i]])

            # To find the most common labels in list closest_labels.
            # Store this labels in predict[i]
            labels = {}
            for label in np.sort(closest_labels):
                if label in labels:
                    labels[label] += 1
                else:
                    labels[label] = 0
            predict[i] = list(a.keys())[np.argmax(list(a.values()))]
        return predict
