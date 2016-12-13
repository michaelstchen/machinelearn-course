import numpy as np
import math
import csv
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import normalize
from scipy.stats import multivariate_normal

def benchmark(pred_labels, true_labels):
    errors = pred_labels != true_labels
    err_rate = sum(errors) / float(len(true_labels))
    indices = errors.nonzero()
    return err_rate, indices

print("Initializations")
execfile('featurize.py')
train_data = np.array(file_dict['training_data'])
train_labels = np.array(file_dict['training_labels'])
test_data = np.array(file_dict['test_data'])

feat_num = train_data.shape[1]

random_state = np.random.get_state()
np.random.shuffle(train_data)
np.random.set_state(random_state)
np.random.shuffle(train_labels)

num_train = train_data.shape[0]
num_test = test_data.shape[0]

set_size = [num_train]
pred_labels = np.zeros(num_test)
for si in range(0, len(set_size)):
    sz = set_size[si]
    print("Slicing, Sorting, Reshaping")
    train_slice = train_data[0:sz]
    label_slice = train_labels[0:sz]

    index = label_slice.argsort()
    t_sort = train_slice[index[::1]]
    t_sort = np.transpose(t_sort)
    l_sort = label_slice[index[::1]]
    
    priors = np.zeros(2)
    for i in l_sort:
        priors[i] = priors[i] + 1
            
    digit_sect = np.zeros(3)
    for i in range(0, 2):
        digit_sect[i+1] = digit_sect[i]+priors[i]
        
    priors = priors / len(l_sort)

    print("Covariance Matrix")
    covmat = np.zeros((2,feat_num,feat_num))
    mu = np.zeros((2, feat_num))
    for i in range(0,2):
        ai = digit_sect[i]
        ae = digit_sect[i+1]
        covmat[i,:,:] = np.cov(t_sort[:, ai:ae])
        mu[i, :] = np.mean(t_sort[:, ai:ae], axis=1)

    a = 0
    overall = np.mean(covmat, axis=0)

    print("Predicting...")
    test_data = np.transpose(test_data)
    for i in range(0, len(pred_labels)):
        x = test_data[:, i]
        max_c = 0
        max_disc = -9999999
        for c in range(0, 2):
            curr = multivariate_normal.logpdf(x,
                                              mean=mu[c],
                                              cov=overall)        
            if curr > max_disc:
                max_c = c
                max_disc = curr
        pred_labels[i] = max_c

with open('test_labels.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(['Id'] + ['Category'])
    for i in range(0, len(pred_labels)):
        writer.writerow([i+1] + [int(pred_labels[i])])
