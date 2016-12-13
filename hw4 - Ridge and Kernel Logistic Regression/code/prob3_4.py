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

def log_regr_kernel2(Xt, yt, Xv, yv, Kt, Kv, a, eps, lam, lim):
    trisks = []
    vrisks = []
    for i in range(0, lim):
        j = np.random.randint(0, len(Xt))
        Kta_j = np.dot(Kt[j], a)
        a = grad_desc_stoch(j, yt[j], Kta_j, a, eps, lam)
        if (i % 100 == 0):
            trisks.append(risk(Kt, a, yt))
            vrisks.append(risk(Kv, a, yv))
    return a, trisks, vrisks

''' LOADING AND SHUFFLING DATA '''
spam_data = loadmat(file_name="spam_data48.mat", mat_dtype=True)
train_data = np.array(spam_data['training_data'])
train_labels = np.transpose(np.array(spam_data['training_labels']))
train_labels = train_labels[:, 0]

random_state = np.random.get_state()
np.random.shuffle(train_data)
np.random.set_state(random_state)
np.random.shuffle(train_labels)

train_data = scale(train_data)
num_train = len(train_data)


''' 6-FOLD CROSS VALIDATION '''
lim = 10000
lam = 0.001
eps = 0.01
# eps = 1e-5
rho_list = [1000, 100, 10, 1, 0.1, 0.01, 0.001]
rho_risks = [0] * len(rho_list)
for i in range(0, len(rho_list)):
    print('Iteration: ' + str(i))
    rho = rho_list[i]
    for k in range(0, num_train, num_train/6):
        td = np.vstack((train_data[:k], train_data[k+num_train/6:]))
        tl = np.append(train_labels[:k],train_labels[k+num_train/6:])
        vd = train_data[k:k+num_train/6]
        vl = train_labels[k:k+num_train/6]
        
        K = kernel_mat(td, td, 1, rho)
        #K = kernel_mat(td, td, 2, rho)
        a = np.zeros(len(td))
        a, risks = log_regr_kernel(td, tl, K, a, eps, lam, lim)
        
        K_val = kernel_mat(vd, td, 1, rho)
        #K_val = kernel_mat(vd, td, 2, rho)        
        rho_risks[i] = rho_risks[i] + risk(K_val, a, vl)
        
    rho_risks[i] = rho_risks[i] / 6

    
rho = rho_list[rho_risks.index(min(rho_risks))]


''' PLOTTING TRAINING AND VALIDATION RISKS '''
k = len(train_data) * 2 / 3
valid_data = train_data[k:]
valid_labels = train_labels[k:]
train_data = train_data[0:k]
train_labels = train_labels[0:k]

lim = 20000
lam = 0.001
a0 = np.zeros(len(train_data))

#eps = 0.01
#rho = 1
#K_train = kernel_mat(train_data, train_data, 1, rho)
#K_valid = kernel_mat(valid_data, train_data, 1, rho)

eps = 1e-5
rho = 100
K_train = kernel_mat(train_data, train_data, 2, rho)
K_valid = kernel_mat(valid_data, train_data, 2, rho)

a, trisks, vrisks = log_regr_kernel2(train_data, train_labels, valid_data, valid_labels, K_train, K_valid, a0, eps, lam, lim)


plt.subplot(2, 1, 1)
plt.plot(trisks)
#plt.title('Linear Kernel')
plt.title('Quadratic Kernel')
plt.ylabel('Training Risk')

plt.subplot(2, 1, 2)
plt.plot(vrisks)
plt.xlabel('Iterations (x100)')
plt.ylabel('Validation Risk')

plt.show()
