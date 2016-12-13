import numpy as np
from math import log, exp

X = np.array([[0, 3, 1],
              [1, 3, 1],
              [0, 1, 1],
              [1, 1, 1]], dtype=np.float)

y = np.array([1, 1, 0, 0], dtype=np.float)
w0 = np.array([-2, 1, 0], dtype=np.float)
eps = 1

def sig(x):
    return 1 / (1 + np.exp(-x))

def mu(X, w):
    return sig(np.dot(X, w))

def risk(X, y, w):
    emp_risk = 0.0
    for i in range(0, len(X)):
        a = y[i]*log(sig(np.dot(w, X[i])))
        b = (1 - y[i])*log(1 - sig(np.dot(w, X[i])))
        emp_risk -= a + b
    return emp_risk
    
def grad_asc_update(X, y, w):
    upd = np.zeros(3)
    for i in range(0, np.size(X, 0)):
        upd += (y[i] - mu(X[i, :], w)) * X[i, :]
    return w + (eps * upd)


R0 = risk(X, y, w0)
mu0 = mu(X, w0)

w1 = grad_asc_update(X, y, w0)
R1 = risk(X, y, w1)
mu1 = mu(X, w1)

w2 = grad_asc_update(X, y, w1)
R2 = risk(X, y, w2)

