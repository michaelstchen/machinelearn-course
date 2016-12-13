import numpy as np
import math
import csv
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import normalize

#benchmark.m, converted
def benchmark(pred_labels, true_labels):
    errors = pred_labels != true_labels
    err_rate = sum(errors) / float(len(true_labels))
    indices = errors.nonzero()
    return err_rate, indices


train_set = loadmat(file_name="train.mat", mat_dtype=True)
test_set = loadmat(file_name="test.mat")

train_data = np.transpose(train_set["train_images"])
train_labels = train_set["train_labels"].reshape(1,-1)[0]
test_data = test_set["test_images"]

random_state = np.random.get_state()
np.random.shuffle(train_data)
np.random.set_state(random_state)
np.random.shuffle(train_labels)

num_train = train_data.shape[0]
num_test = test_data.shape[0]

for i in range(0, num_train):
    train_data[i] = train_data[i] / np.linalg.norm(train_data[i])
    
set_size = [100, 200, 500, 1000, 2000, 5000, 10000, 30000, 50000]
err_list = [0, 0, 0, 0, 0, 0, 0, 0, 0]
for si in range(0, len(set_size)):
    sz = set_size[si]
    train_slice = train_data[0:sz]
    label_slice = train_labels[0:sz]

    index = label_slice.argsort()
    t_sort = train_slice[index[::1]]
    l_sort = label_slice[index[::1]]
    t_flat = t_sort.reshape(len(t_sort), -1)
    t_flat = np.transpose(t_flat)
    for i in range(0, sz):
        t_flat[:, i] = t_flat[:, i] / np.linalg.norm(t_flat[:, i])

    priors = np.zeros(10)
    for i in l_sort:
        priors[i] = priors[i] + 1
            
    digit_sect = np.zeros(11)
    for i in range(0, 10):
        digit_sect[i+1] = digit_sect[i]+priors[i]
        
    priors = priors / len(l_sort)
            
    covmat = np.zeros((10,784,784))
    mu = np.zeros((10, 784))
    for i in range(0,10):
        ai = digit_sect[i]
        ae = digit_sect[i+1]
        covmat[i,:,:] = np.cov(t_flat[:, ai:ae])
        mu[i, :] = np.mean(t_flat[:, ai:ae], axis=1)

    n = 10000
    overall = np.mean(covmat, axis=0)
    train_valid = train_data[len(train_data)-n:len(train_data)]
    train_valid = train_valid.reshape(len(train_valid), -1)
    train_valid = np.transpose(train_valid)
    labels_valid = train_labels[len(train_data)-n:len(train_data)]
    preds = np.zeros(len(labels_valid))

    a = 0.001
    over_inv = np.linalg.inv(overall + a*np.identity(784))
    firsts = np.zeros((10, 784))
    seconds = np.zeros(10)
    thirds = np.zeros(10)
    for c in range(0,10):
        mu_t = np.transpose(mu[c])
        firsts[c] = np.dot(mu_t, over_inv)
        seconds[c] = 0.5 * np.dot(mu_t, np.dot(over_inv,mu[c]))
        thirds[c] = np.log(priors[c])

    for i in range(0, len(preds)):
        x = train_valid[:, i]
        max_c = 0
        max_disc = 0
        for c in range(0, 10):
            mu_t = np.transpose(mu[c])
            first = np.dot(firsts[c],x)
            curr = first - seconds[c] + thirds[c]
            if curr > max_disc:
                max_c = c
                max_disc = curr
        preds[i] = max_c        
    err_list[si], ind = benchmark(preds, labels_valid)

plt.plot(set_size, err_list)
plt.title('Error Rates for LDA')
plt.xlabel('Number Training Samples')
plt.ylabel('Error Rate')
plt.show()
