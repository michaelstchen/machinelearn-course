import numpy as np
import math
import csv
import matplotlib.pyplot as plt
from scipy.io import loadmat

def benchmark(pred_labels, true_labels):
    errors = pred_labels != true_labels
    err_rate = sum(errors) / float(len(true_labels))
    indices = errors.nonzero()
    return err_rate, indices

housing_data = loadmat(file_name="housing_data.mat", mat_dtype=True)

Xtrain = housing_data["Xtrain"]
Ytrain = housing_data["Ytrain"]
Xvalid = housing_data["Xvalidate"]
Yvalid = housing_data["Yvalidate"]

Ytrain = Ytrain[:, 0]
Yvalid = Yvalid[:, 0]

# Centering data
Xtrain = Xtrain - np.mean(Xtrain, axis=0)
Xvalid = Xvalid - np.mean(Xvalid, axis=0)

num_train = Xtrain.shape[0]
num_valid = Xvalid.shape[0]

param_list = [0.0, 1e-10, 1e-5, 1e-3, 1e-2, 1, 1e2, 1e3, 1e5, 1e10, 1e15, 1e20]
#J_list = [0] * len(param_list)
R_list = [0] * len(param_list)
for i in range(0, len(param_list)):
    for k in range(0, num_train, num_train/10):
        param = param_list[i]
        X = np.vstack((Xtrain[:k], Xtrain[k+num_train/10:]))
        Y = np.append(Ytrain[:k], Ytrain[k+num_train/10:])
        Xval = Xtrain[k:k+num_train/10]
        Yval = Ytrain[k:k+num_train/10]
        
        Xt_X = np.dot(X.T, X)
        lambda_I = param * np.identity(len(Xt_X))
        Xt_y = np.dot(X.T, Y)
        w = np.dot(np.linalg.inv(Xt_X + lambda_I), Xt_y)

        alpha = np.mean(Y)
        Ypred = np.dot(Xval, w) + alpha*np.ones(len(Yval))
        res = Ypred - Yval
        #J_list[i] = J_list[i] + np.dot(res, res)+param*np.dot(w, w)
        R_list[i] = R_list[i] + np.dot(res, res)
        
    #J_list[i] = J_list[i] / 10
    R_list[i] = R_list[i] / 10

param = param_list[R_list.index(min(R_list))]
Xt = np.transpose(Xtrain)
Xt_X = np.dot(Xt, Xtrain)
lambda_I = param * np.identity(len(Xt_X))
Xt_y = np.dot(Xt, Ytrain)
w = np.dot(np.linalg.inv(Xt_X + lambda_I), Xt_y)

alpha = np.mean(Ytrain)
Ypred = np.dot(Xvalid, w) + alpha*np.ones(len(Yvalid))
res_valid = Yvalid - Ypred
RSS = np.dot(res_valid, res_valid)

plt.scatter(range(0, len(w)), w)
plt.title('Regression Weights')
plt.xlabel('Index')
plt.ylabel('Value')
plt.show()
