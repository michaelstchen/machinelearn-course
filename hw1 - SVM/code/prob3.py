import numpy as np
import math
import csv
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import metrics, svm

#benchmark.m, converted
def benchmark(pred_labels, true_labels):
    errors = pred_labels != true_labels
    err_rate = sum(errors) / float(len(true_labels))
    indices = errors.nonzero()
    return err_rate, indices

train_set = loadmat(file_name="train.mat")
test_set = loadmat(file_name="test.mat")

train_data = np.transpose(train_set["train_images"])
train_labels = train_set["train_labels"].reshape(1,-1)[0]
test_data = test_set["test_images"]

num_train = train_data.shape[0]
num_test = test_data.shape[0]

train_data_flat = train_data.reshape(num_train, -1)
test_data_flat = test_data.reshape(num_test, -1)

random_state = np.random.get_state()
np.random.shuffle(train_data)
np.random.set_state(random_state)
np.random.shuffle(train_labels)

c_list = [0.0000001, 0.000001, 0.00001]
xc_err = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for c in range(0,len(c_list)):
    print("Current C index: " + str(c))
    for k in range(0,10000,1000):
        print("k: " + str(k))
        cv_train = np.vstack((train_data_flat[:k, :], \
                              train_data_flat[k+1000:10000, :]))
        cv_train_labels = np.append(train_labels[:k],\
                                    train_labels[k+1000:10000])
        cv_valid = train_data_flat[k:k+1000, :]
        cv_valid_labels = train_labels[k:k+1000]
        
        clf = svm.SVC(C=c_list[c], kernel='linear')
        clf.fit(cv_train, cv_train_labels)
        
        pred_labels = clf.predict(cv_valid)
        err_rate, indices = benchmark(pred_labels, cv_valid_labels)
        c_err[c] = c_err[c] + (1-err_rate)
    print("\n")
    c_err[c] = c_err[c] / 10

with open('prob3_crossvalid.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(['C Value'] + ['% Valid'])
    for i in range(0, len(c_list)):
        writer.writerow([c_list[i]] + [c_err[i]])
    
c_optimal = c_list[c_err.index(max(c_err))]

clf = svm.SVC(C=c_optimal, kernel='linear')
clf.fit(train_data_flat[:10000], train_labels[:10000])

pred_test_labels = clf.predict(test_data_flat)

with open('test_labels.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(['Id'] + ['Category'])
    for i in range(0, len(pred_test_labels)):
        writer.writerow([i+1] + [pred_test_labels[i]])
