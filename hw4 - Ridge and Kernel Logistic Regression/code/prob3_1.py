import numpy as np
import math
import csv
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import scale, binarize

housing_data = loadmat(file_name="spam_data.mat", mat_dtype=True)
train_data = np.array(housing_data['training_data'])
train_labels = np.transpose(np.array(housing_data['training_labels']))

train_data_i = scale(train_data)
train_data_ii = np.log(train_data + 0.1)
train_data_iii = binarize(train_data)

def sig(x):
    return 1 / (1 + np.exp(-x))

def risk(X, y, w):
    sig_Xw = sig(np.dot(X, w))
    a = np.multiply(y, np.log(sig_Xw + 1e-100))
    b = np.multiply(1-y, np.log(1-sig_Xw + 1e-100))
    emp_risk = np.sum(a + b)
    return -emp_risk

def grad_desc_batch(X, y, w, eps):
    upd = y - sig(np.dot(X, w))
    upd = (X.T * upd).T
    upd = np.sum(upd, axis=0)
    return w + (eps * upd)

def log_regr_batch(X, y, w0, learn, err):
    risk_list = []
    risk_list.append(risk(X, y, w0))
    w = grad_desc_batch(X, y, w0, learn)
    risk_list.append(risk(X, y, w))
    i = 2
    while (abs(risk_list[-1]-risk_list[-2]) > err):
        w = grad_desc_batch(X, y, w, learn)
        if (i % 10 == 0):
            risk_list.append(risk(X, y, w))
        i = i + 1
    return risk_list


w0 = np.ones(len(train_data[0])) / 100

# For preprocessing (i)
risk_i = log_regr_batch(train_data_i,train_labels[:,0],w0,1e-3,0.1)

# For preprocessing (ii)
risk_ii = log_regr_batch(train_data_ii,train_labels[:,0],w0,1e-6,0.1)

# For preprocessing (iii)
risk_iii = log_regr_batch(train_data_iii,train_labels[:,0],w0,1e-3,0.1)

plt.subplot(3, 1, 1)
plt.plot(range(0,len(risk_i)), risk_i)
plt.title('method (i)')
plt.ylabel('Risk')

plt.subplot(3, 1, 2)
plt.plot(range(0,len(risk_ii)), risk_ii)
plt.title('method (ii)')
plt.ylabel('Risk')

plt.subplot(3, 1, 3)
plt.plot(range(0,len(risk_iii)), risk_iii)
plt.title('method (iii)')
plt.ylabel('Risk')
plt.xlabel('Iterations (x10)')

plt.show()
