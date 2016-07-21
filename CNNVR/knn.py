#!/usr/bin/env python

from data.data_utils import load_CIFAR10
from classifiers import KNearestNeighbor
import numpy as np
import matplotlib.pyplot as plt
import time

# Load the raw CIFAR-10 data.
CIFA_PATH = 'data/cifar-10'
train_dataset, train_labels, test_dataset, test_labels = load_CIFAR10(CIFA_PATH)

org_test_dataset = test_dataset
org_train_dataset = train_dataset

print('Training dataset shape:', train_dataset.shape)
print('Training labels shape', train_labels.shape)
print('Test dataset shape:', test_dataset.shape)
print('Test labels shape:', test_labels.shape)


# Visualize some examples from the dataset
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship'
        , 'truck']
num_classes = len(classes)
samples_per_class = 7
for label, cls in enumerate(classes):
    id_train_dataset = np.flatnonzero(train_labels == label)
    id_train_dataset = np.random.choice(id_train_dataset, samples_per_class, replace=False)
    for i, id_train in enumerate(id_train_dataset):
        plt_id_train = i * num_classes + label + 1
        plt.subplot(samples_per_class, num_classes, plt_id_train)
        plt.imshow(train_dataset[id_train].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
# plt.ion()
plt.show()

# Subsample the data for more efficient code execution in this exercise
NUM_TRAININ = 5000
mask = range(NUM_TRAININ)
train_dataset = train_dataset[mask]
train_labels = train_labels[mask]

NUM_TEST = 500
mask = range(NUM_TEST)
test_dataset = test_dataset[mask]
test_labels = test_labels[mask]

# Reshape the image data into rows
train_dataset = np.reshape(train_dataset, (train_dataset.shape[0], -1))
test_dataset = np.reshape(test_dataset, (test_dataset.shape[0], -1))
print('Reshpaed train_dataset.shape:', train_dataset.shape, ', test_dataset.shape:', test_dataset.shape)

# Create a kNN classifiter instance.
# Remember that training a kNN classifier is a noop:
# the Classifier simply rememvers the data and does no further processing
classifier = KNearestNeighbor()
classifier.train(train_dataset, train_labels)


# Test compute_distances_two_loops:
# start_time = time.time()
# dists_two_loops = classifier.compute_distances_two_loops(test_dataset)
# print("--- KNN one loop execution time: {0} seconds ---".format((time.time() - start_time)))
# print('Distance shape:', dists_two_loops.shape)

# Test compute_distances_one_loop:
# start_time = time.time()
# dists_one_loop = classifier.compute_distances_one_loop(test_dataset)
# print("--- KNN two loops execution time: {0} seconds ---".format((time.time() - start_time)))
# print('Distance shape:', dists_one_loop.shape)


# Test compute_distances_no_loop:
start_time = time.time()
dists_no_loop = classifier.compute_distances_no_loop(test_dataset)
print("--- KNN no loop execution time: {0} seconds ---".format((time.time() - start_time)))
print('Distance shape:', dists_no_loop.shape)

# Visualize the distance matrix: each row is a single test example and its
# distances to training examples
plt.imshow(dists_no_loop, interpolation='none')
# plt.ion()
plt.show()

# Now implement the function predict_labels and run the code below:
# Use k = 1
test_labels_predict = classifier.predict_labels(dists_no_loop, k = 1)

# Compute and print the fraction of correctly predicted examples

# for i in [0, 1, 2 ,3 ,4]:
#     print(test_labels_predict[i], classes[int(test_labels_predict[i])], test_labels[i], classes[test_labels[i]])

num_correct = np.sum(test_labels_predict == test_labels)
accuracy = float(num_correct) / NUM_TEST
print('Got {0:d} / {1:d} correct => accuracy: {2:f}'.format(num_correct, NUM_TEST, accuracy))

index = -1
for j in range(14):
    plt.subplot(samples_per_class, 2, j + 1)
    if j % 2 == 0:
        index += 1
        plt.title(classes[int(test_labels_predict[index])])
    else:
        plt.title(classes[(test_labels[index])])
    plt.imshow(org_test_dataset[index].astype('uint8'))
    plt.axis('off')
# plt.ion()
plt.show()


# Cross-Validation
NUM_FOLDES = 5
K_CHOICES = [1, 3, 5, 8 ,10 , 12, 15, 20, 50, 100]

train_dataset_foldes = []
train_labels_foldes = []

train_dataset_foldes = np.array_split(train_dataset, NUM_FOLDES)
train_labels_foldes = np.array_split(train_labels, NUM_FOLDES)

# This dictionary is holding the accuracies for different values for k that
# we find when running cross-validation.
k_to_accuracies = {}


for k in K_CHOICES:
    accuracy = []
    for i in range(NUM_FOLDES):
        train_dataset_tmp = []
        train_labels_tmp = []
        test_dataset_tmp = []
        test_labels_tmp = []
        for j in range(NUM_FOLDES):
            if i == j:
                test_dataset_tmp = train_dataset_foldes[j]
                test_labels_tmp = train_labels_foldes[j]
                continue
            train_dataset_tmp.append(train_dataset_foldes[j])
            train_labels_tmp.append(train_labels_foldes[j])
        train_dataset_tmp = np.vstack(train_dataset_tmp)
        train_labels_tmp = np.concatenate(train_labels_tmp)

        classifier_cv = KNearestNeighbor()
        classifier_cv.train(train_dataset_tmp, train_labels_tmp)
        dists = classifier_cv.compute_distances_no_loop(test_dataset_tmp)

        test_labels_predict = classifier_cv.predict_labels(dists, k)
        
        # Compute and print the fraction of correctly predicted examples
        num_correct = np.sum(test_labels_predict == test_labels_tmp)
        acc = float(num_correct) / len(test_labels_tmp)
        accuracy.append(acc)
    k_to_accuracies[k] = accuracy


# Print out the computed accuracies
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k {0:d}, accuracy = {1:f}'.format(k, accuracy))


# Plot the raw observations
for k in K_CHOICES:
    accuracies = k_to_accuracies[k]
    plt.scatter([k] * len(accuracies), accuracies)

# Plot the trend line with error bars that correspond to standard devistion
accuracies_mean = np.array([np.mean(v) for k, v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k, v in sorted(k_to_accuracies.items())])
plt.errorbar(K_CHOICES, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()


# Based on the cross-validation results above, choose the best value for k,
# retrain the classifier using all the training data, and test it on the test
# data.
BEST_K = 7

classifier = KNearestNeighbor()
classifier.train(train_dataset, train_lables)
test_labels_pred = classifier.predict(test_data, k=BEST_K)

# Compute and display the accuracy
num_correct = np.sum(test_labels_pred == test_labels)
accuracy = float(num_correct) / NUM_TEST
print('Got {0:d} / {1:d} correct => accuracy: {2:f}'.format(num_correct, NUM_TEST, accuracy))
