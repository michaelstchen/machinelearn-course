import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi

def multgaussprob(x, y, mu, covmat):
    prob_a = np.zeros((len(x), len(y)))
    norm = 1 / (sqrt(2*pi)*np.linalg.det(covmat))
    
    for ix in range(0, len(x)):
        for iy in range(0, len(y)):
            x_right = np.array([[x[ix]],[y[iy]]]) - mu
            x_left = np.transpose(x_right)
            x_exp = np.dot(x_left, np.dot(np.linalg.inv(covmat),
                                          x_right))
            prob_a[ix, iy] = norm * np.exp(x_exp/-2)

    return prob_a
    

x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)

#part (a)
mu_a = np.array([[1], [1]])
covmat_a = np.array([[2, 0], [0, 1]])
prob_a = multgaussprob(x, y, mu_a, covmat_a)
ax_a = plt.subplot(2, 3, 1)
ax_a.set_title("part (a)")
ax_a.set_xlabel("x")
ax_a.set_ylabel("y")
ax_a.contour(x, y, prob_a, 10)

#part (b)
mu_b = np.array([[-1], [2]])
covmat_b = np.array([[3, 1], [1, 2]])
prob_b = multgaussprob(x, y, mu_b, covmat_b)
ax_b = plt.subplot(2, 3, 2)
ax_b.set_title("part (b)")
ax_b.set_xlabel("x")
ax_b.set_ylabel("y")
ax_b.contour(x, y, prob_b, 10)

#part (c)
mu_c1 = np.array([[0], [2]])
mu_c2 = np.array([[2], [0]])
covmat_c = np.array([[1, 1], [1, 2]])
prob_c1 = multgaussprob(x, y, mu_c1, covmat_c)
prob_c2 = multgaussprob(x, y, mu_c2, covmat_c)
prob_c = prob_c1 - prob_c2
ax_c = plt.subplot(2, 3, 3)
ax_c.set_title("part (c)")
ax_c.set_xlabel("x")
ax_c.set_ylabel("y")
ax_c.contour(x, y, prob_c)

#part (d)
mu_d1 = np.array([[0], [2]])
mu_d2 = np.array([[2], [0]])
covmat_d1 = np.array([[1, 1], [1, 2]])
covmat_d2 = np.array([[3, 1], [1, 2]])
prob_d1 = multgaussprob(x, y, mu_d1, covmat_d1)
prob_d2 = multgaussprob(x, y, mu_d2, covmat_d2)
prob_d = prob_d1 - prob_d2
ax_d = plt.subplot(2, 3, 4)
ax_d.set_title("part (d)")
ax_d.set_xlabel("x")
ax_d.set_ylabel("y")
ax_d.contour(x, y, prob_d)

#part (e)
mu_e1 = np.array([[1], [1]])
mu_e2 = np.array([[-1], [-1]])
covmat_e1 = np.array([[1, 0], [0, 2]])
covmat_e2 = np.array([[2, 1], [1, 2]])
prob_e1 = multgaussprob(x, y, mu_e1, covmat_e1)
prob_e2 = multgaussprob(x, y, mu_e2, covmat_e2)
prob_e = prob_e1 - prob_e2
ax_e = plt.subplot(2, 3, 5)
ax_e.set_title("part (e)")
ax_e.set_xlabel("x")
ax_e.set_ylabel("y")
ax_e.contour(x, y, prob_e)

plt.show()
