import numpy as np
import math
import csv
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import svm

#benchmark.m, converted
def benchmark(pred_labels, true_labels):
    errors = pred_labels != true_labels
    err_rate = sum(errors) / float(len(true_labels))
    indices = errors.nonzero()
    return err_rate, indices

execfile('featurize.py')
train_data = np.array(file_dict['training_data'])
train_labels = np.array(file_dict['training_labels'])
test_data = np.array(file_dict['test_data'])

random_state = np.random.get_state()
np.random.shuffle(train_data)
np.random.set_state(random_state)
np.random.shuffle(train_labels)

num_msg = len(train_data)
c_list = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
c_err = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for c in range(0, len(c_list)):
    print("Current C index: " + str(c))
    for k in range(0, num_msg, num_msg/10):
        print("k: " + str(k))
        cv_train = np.vstack((train_data[:k], \
                              train_data[k+num_msg/10:]))
        cv_train_labels = np.append(train_labels[:k],\
                                    train_labels[k+num_msg/10:])
        cv_valid = train_data[k:k+num_msg/10]
        cv_valid_labels = train_labels[k:k+num_msg/10]
        
        clf = svm.LinearSVC(C=c_list[c])
        clf.fit(cv_train, cv_train_labels)
        
        pred_labels = clf.predict(cv_valid)
        err_rate, indices = benchmark(pred_labels, cv_valid_labels)
        c_err[c] = c_err[c] + (1-err_rate)
    print("\n")
    c_err[c] = c_err[c] / 10


with open('prob4_crossvalid.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(['C Value'] + ['% Valid'])
    for i in range(0, len(c_list)):
        writer.writerow([c_list[i]] + [c_err[i]])

c_optimal = c_list[c_err.index(max(c_err))]

clf = svm.LinearSVC(C=c_optimal)
clf.fit(train_data, train_labels)

pred_labels = clf.predict(test_data)


with open('test_labels.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(['Id'] + ['Category'])
    for i in range(0, len(pred_labels)):
        writer.writerow([i+1] + [pred_labels[i]])
