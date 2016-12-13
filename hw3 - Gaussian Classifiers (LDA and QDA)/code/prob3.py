import numpy as np
import matplotlib.pyplot as plt
import math
from math import sqrt, pi

N = 100
samples = np.zeros((N, 2))
for i in range(0, N):
    x1 = np.random.normal(3, 3)
    x2 = x1/2.0 + np.random.normal(4, 2)
    samples[i, :] = np.array([x1, x2])

#part(a)
mu = np.mean(samples, axis=0)

#part(b)
covmat = np.cov(np.transpose(samples))

#part(c)
eigvals, eigvecs = np.linalg.eig(covmat)

#part(d)
plt.figure()
plt.scatter(samples[:,0], samples[:,1], color='blue')
plt.scatter(mu[0], mu[1], color='red')
ang0 = math.atan(eigvecs[1,0]/eigvecs[0,0])
ang1 = math.atan(eigvecs[1,1]/eigvecs[0,1])
plt.arrow(mu[0], mu[1],
          eigvals[0]*math.cos(ang0),eigvals[0]*math.sin(ang0))
plt.arrow(mu[0], mu[1],
          eigvals[1]*math.cos(ang1),eigvals[1]*math.sin(ang1))
plt.title('part (d)')
plt.xlabel('X1')
plt.ylabel('X2')
plt.xlim([-15, 15])
plt.ylim([-15, 15])


#part(e)
trans_samples = np.zeros((N, 2))
for i in range(0, N):
    xcent = np.transpose(samples[i, :]-mu)
    xrot = np.dot(np.transpose(eigvecs), np.transpose(xcent))
    trans_samples[i, :] = np.transpose(xrot)
    
plt.figure()
plt.scatter(trans_samples[:,0], trans_samples[:,1], color='blue')
plt.scatter(0, 0, color='red')
plt.title('part (e)')
plt.xlabel('v1')
plt.ylabel('v2')
plt.xlim([-15, 15])
plt.ylim([-15, 15])

plt.show()
