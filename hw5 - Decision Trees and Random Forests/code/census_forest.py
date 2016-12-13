import time
import numpy as np
import math
import DecisionTree as dt
from random import randint
from scipy.io import loadmat

def benchmark(pred_labels, true_labels):
    errors = pred_labels != true_labels
    err_rate = sum(errors) / float(len(true_labels))
    indices = errors.nonzero()
    return err_rate, indices

print("Preprocessing...")
execfile('featurize_census.py')

random_state = np.random.get_state()
np.random.shuffle(train_data)
np.random.set_state(random_state)
np.random.shuffle(train_labels)

k = len(train_data)//2
valid_data = train_data[k:]
valid_labels = train_labels[k:]
train_data_sub = train_data[0:k]
train_labels_sub = train_labels[0:k]

numtrain = k
depth = 50
m = 10
n = 10

print("Generating Random Forest...")
T = 30
tree_list = []
for t in range(0, T):
    print("T = " + str(t))
    before = time.time()
    rind = [randint(0, k-1) for x in range(0, numtrain)]
    dectree = dt.DecisionTree(depth, m, n)
    dectree.train(train_data_sub[rind], train_labels_sub[rind])
    tree_list.append(dectree)
    after = time.time()
    print("Took " + str(after - before) + " seconds\n")

print("Predicting on Validation...")
pred_val_labels = np.zeros(len(valid_labels))
for dt in tree_list:
    pred_val_labels += dt.predict(valid_data)

pred_val_labels = np.round(pred_val_labels / float(T))


err, ind = benchmark(pred_val_labels, valid_labels)
print("error rate: " + str(err))

print("Predicting on Test...")
pred_test_labels = np.zeros(len(test_data))
for dt in tree_list:
    pred_test_labels += dt.predict(test_data)

pred_test_labels = np.round(pred_test_labels / float(T))

print("Writing to CSV")
with open('test_labels.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(['Id'] + ['Category'])
    for i in range(0, len(pred_test_labels)):
        writer.writerow([i+1] + [int(pred_test_labels[i])])
