import numpy as np
import math
import csv
import matplotlib.pyplot as plt
from scipy.io import loadmat

housing_data = loadmat(file_name="housing_data.mat", mat_dtype=True)

Xtrain = housing_data["Xtrain"]
Ytrain = housing_data["Ytrain"]
Xvalid = housing_data["Xvalidate"]
Yvalid = housing_data["Yvalidate"]

num_train = Xtrain.shape[0]
num_valid = Xvalid.shape[0]
Xtrain = np.append(Xtrain, (np.ones(num_train)).reshape(num_train, 1), 1)
Xvalid = np.append(Xvalid, (np.ones(num_valid)).reshape(num_valid, 1), 1)

X_t = np.transpose(Xtrain)
X_pinv = np.linalg.inv(np.dot(X_t, Xtrain))
X_pinv = np.dot(X_pinv, X_t)
w = np.dot(X_pinv, Ytrain)

Ypred = np.dot(Xvalid, w)
res_train = np.dot(Xtrain, w) - Ytrain
res_valid = Yvalid - Ypred
RSS = np.sum(res_valid**2)

minYpred = np.amin(Ypred)
maxYpred = np.amax(Ypred)

plt.figure()
plt.scatter(range(1,9), w[0:8])
plt.title('Regression Weights')
plt.xlabel('Index')
plt.ylabel('Value')
plt.show()


plt.figure()
plt.hist(res_train, 100)
plt.title('Training Residuals')
plt.xlabel('Residual')
plt.ylabel('Count')
plt.show()
