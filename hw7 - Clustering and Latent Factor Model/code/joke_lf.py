import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def benchmark(pred_labels, true_labels):
    errors = pred_labels != true_labels
    err_rate = sum(errors) / float(len(true_labels))
    indices = errors.nonzero()
    return err_rate, indices

# Load joke training data
joke_mat = loadmat(file_name="joke_train.mat", mat_dtype=True)
joke_data = np.array(joke_mat["train"])
joke_data = np.nan_to_num(joke_data)

# Load joke validation data
joke_valid = []
for line in file('validation.txt', 'r'):
    valid_pt = []
    for num in line.strip().split(','):
        valid_pt.append(int(num))
    joke_valid.append(valid_pt)

joke_valid = np.array(joke_valid)

# Use SVD to find principle components
d = 20
U, D, Vt = np.linalg.svd(joke_data, full_matrices=False)
U = U[:, :d]
Vt = Vt[:d, :]

# Calculate mse
mse = np.sum((U.dot(Vt) - joke_data)**2)

# Predictions
pred = [0] * len(joke_valid)
for i in range(0, len(joke_valid)):
    user_ind = joke_valid[i][0] - 1
    joke_ind = joke_valid[i][1] - 1
    score = np.dot(U[user_ind], Vt[:, joke_ind])
    if score > 0:
        pred[i] = 1
    else:
        pred[i] = 0

err, ind = benchmark(joke_valid[:, 2], pred)
