import numpy as np
import math
import csv
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import scale, binarize

def kernel_mat(X1, X2, deg, rho):
    K = np.dot(X1, X2.T) + rho
    return K**deg

def sig(x):
    return 1 / (1 + np.exp(-x))

def risk(K, a, y):
    sig_Ka = sig(np.dot(K, a))
    a = np.multiply(y, np.log(sig_Ka + 1e-100))
    b = np.multiply(1-y, np.log(1 - sig_Ka + 1e-100))
    return -np.sum(a + b)

def grad_desc_stoch(i, y_i, Ka_i, a, eps, lam):
    a_upd = a - (eps * lam) * a
    a_upd[i] = a_upd[i] + eps * (y_i - sig(Ka_i))
    return a_upd
    
def log_regr_kernel(X, y, K, a, eps, lam, lim):
    risks = []
    for i in range(0, lim):
        j = np.random.randint(0, len(X))
        Ka_j = np.dot(K[j], a)
        a = grad_desc_stoch(j, y[j], Ka_j, a, eps, lam)
        if (i % 100 == 0):
            risks.append(risk(K, a, y))
    return a, risks


''' LOADING DATA '''
spam_data = loadmat(file_name="spam_data.mat", mat_dtype=True)
train_data = np.array(spam_data['training_data'])
train_labels = np.transpose(np.array(spam_data['training_labels']))
train_labels = train_labels[:, 0]

train_data = scale(train_data)
train_data = train_data[:3000]
train_labels = train_labels[:3000]
num_train = len(train_data)

''' LINEAR KERNEL LOGISTIC REGRESSION'''
lim = 15000
lam = 0.001
a0 = np.zeros(len(train_data))

eps = 1e-5
rho = 100
K_train = kernel_mat(train_data, train_data, 2, rho)

a, risks = log_regr_kernel(train_data, train_labels, K_train, a0, eps, lam, lim)

''' PREDICTING '''
test_data = np.array(spam_data['test_data'])
test_data = scale(test_data)
K_test = kernel_mat(test_data, train_data, 1, rho)
pred_labels = np.rint(sig(np.dot(K_test, a)))

with open('test_labels.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(['Id'] + ['Category'])
    for i in range(0, len(pred_labels)):
        writer.writerow([i+1] + [int(pred_labels[i])])

