import numpy as np
import math
import csv
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import scale, binarize

spam_data = loadmat(file_name="spam_data.mat", mat_dtype=True)
train_data = np.array(spam_data['training_data'])
train_labels = np.transpose(np.array(spam_data['training_labels']))

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

def grad_desc_stoch(X, y, w, eps):
    upd = (y - sig(np.dot(X, w))) * X
    return w + (eps * upd)

def log_regr_stoch(X, y, w, learn, lim):
    risk_list = []
    for t in range(1, lim):
        j = np.random.randint(0, len(X)-1)
        w = grad_desc_stoch(X[j], y[j], w, learn/t)
        if (t % 10 == 0):
            risk_list.append(risk(X, y, w))
    return risk_list

lim = 5000
w0 = np.zeros(len(train_data[0]))
w0[0] = 1.0
risk_i = log_regr_stoch(train_data_i,train_labels[:,0],w0,0.01,lim)
risk_ii = log_regr_stoch(train_data_ii,train_labels[:,0],w0,0.01,lim)
risk_iii = log_regr_stoch(train_data_iii,train_labels[:,0],w0,0.01,lim)

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
plt.xlabel('Iterations (x10)')
plt.ylabel('Risk')

plt.show()

