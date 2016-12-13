import time
import numpy as np
import DecisionTree as dt

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

k = len(train_data) * 2 // 3
valid_data = train_data[k:]
valid_labels = train_labels[k:]
train_data_sub = train_data[0:k]
train_labels_sub = train_labels[0:k]

depth = 10
m = len(train_data[0])
n = 100
print("Building Decision Tree")
before = time.time()
dectree = dt.DecisionTree(depth, m, n)
dectree.train(train_data_sub, train_labels_sub)
after = time.time()
print("Training took " + str(after - before) + " seconds")

print("Predicting")
before = time.time()
pred_labels = dectree.predict(valid_data)
err, ind = benchmark(pred_labels, valid_labels)
after = time.time()
print("Predicting took " + str(after - before) + " seconds")

