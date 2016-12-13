# Script for formatting inputs and checking accuracies for
# Neural Net using MSE Loss function

import csv
import time
import neuralnets as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def benchmark(pred_labels, true_labels):
    errors = pred_labels != true_labels
    err_rate = sum(errors) / float(len(true_labels))
    indices = errors.nonzero()
    return err_rate, indices

def vector2digit(v):
    d = np.zeros((len(v), 1))
    for i in range(0, len(v)):
        d[i] = np.argmax(v[i])
    return d

# Load image data
image_mat = loadmat(file_name="train.mat", mat_dtype=True)

# Flatten image data
image_data_orig = np.array(image_mat['train_images']).T
for i in range(0, len(image_data_orig)):
    image_data_orig[i] = image_data_orig[i].T
image_data = image_data_orig.reshape(len(image_data_orig), -1)

# Normalize images' pixel values
image_data = image_data / 255.0

# Vectorize labels
image_lbls = np.array(image_mat['train_labels'])
image_labels = np.zeros((len(image_lbls), 10))
for i in range(0, len(image_lbls)):
    ind = int(image_lbls[i])
    image_labels[i][ind] = 1.0

# Shuffling data
np.random.seed(0)
random_state = np.random.get_state()
np.random.shuffle(image_data)
np.random.set_state(random_state)
np.random.shuffle(image_labels)

# Partitioning data into training and validation sets
k = 50000
valid_data = image_data[k:]
valid_labels = image_labels[k:]
train_data = image_data[0:k]
train_labels = image_labels[0:k]

# Training Neural Net
before = time.time()
digits = nn.NeuralNet(784, 200, 10, 0)
digits.initialize_weights()
loss, acc = digits.trainNeuralNet(train_data, train_labels, 0.01, 1)
after = time.time()
print("TOOK " + str(after-before) + " SECONDS")

# Saving Weights to File
np.save('weights_mse_W1', digits.W1)
np.save('weights_mse_W2', digits.W2)
np.save('mse_acc', np.array(acc))
np.save('mse_loss', np.array(loss))

# Predicting on Validaton Set
valid_preds = digits.predictNeuralNet(valid_data)
valid_pred_labels = vector2digit(valid_preds)
valid_labels = vector2digit(valid_labels)
err, ind = benchmark(valid_pred_labels, valid_labels)
print("VALIDATION ERROR: " + str(err))

