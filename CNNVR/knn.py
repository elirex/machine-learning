#!/usr/bin/env python

from data.data_utils import load_CIFAR10
import numpy as np
import matplotlib.pyplot as plt

# Load the raw CIFAR-10 data.
CIFA_PATH = 'data/cifar-10'
train_dataset, train_labels, test_dataset, test_labels = load_CIFAR10(CIFA_PATH)

print('Training dataset shape:', train_dataset.shape)
print('Training labels shape', train_labels.shape)
print('Test dataset shape:', test_dataset.shape)
print('Test labelk shape:', test_labels.shape)

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
plt.ion()
plt.show()
print('Finished')
