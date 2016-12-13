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

# Load joke validation data
joke_valid = []
for line in file('validation.txt', 'r'):
    valid_pt = []
    for num in line.strip().split(','):
        valid_pt.append(int(num))
    joke_valid.append(valid_pt)

joke_valid = np.array(joke_valid)

# Calculating avg score for each joke
joke_avg_score = np.nanmean(joke_data, 0)

# Predicting using just the average scores
pred = [0] * len(joke_valid)
for i in range(0, len(joke_valid)):
    pred[i] = joke_avg_score[joke_valid[i][1] - 1]
    if pred[i] > 0:
        pred[i] = 1
    else:
        pred[i] = 0

err, ind = benchmark(joke_valid[:, 2], pred)
