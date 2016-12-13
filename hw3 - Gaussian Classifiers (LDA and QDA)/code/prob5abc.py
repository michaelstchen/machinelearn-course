import numpy as np
import math
import csv
import matplotlib.pyplot as plt
from scipy.io import loadmat

train_set = loadmat(file_name="train.mat")
test_set = loadmat(file_name="test.mat")

train_data = np.transpose(train_set["train_images"])
train_labels = train_set["train_labels"].reshape(1,-1)[0]
test_data = test_set["test_images"]

num_train = train_data.shape[0]
num_test = test_data.shape[0]

train_data_flat = train_data.reshape(num_train, -1)
train_data_flat = np.transpose(train_data_flat)
test_data_flat = test_data.reshape(num_test, -1)
test_data_flat = np.transpose(test_data_flat)

covmat = np.zeros((10,784,784))
mu = np.zeros((10, 784))
for i in range(0,10):
    ai = 2000 + (i * 6000)
    ae = ai+2000
    covmat[i,:,:] = np.cov(train_data_flat[:, ai:ae])
    mu[i, :] = np.mean(train_data_flat[:, ai:ae], axis=1)

priors = np.zeros(10)
for i in train_labels:
    priors[i] = priors[i] + 1
priors = priors / len(train_labels)

