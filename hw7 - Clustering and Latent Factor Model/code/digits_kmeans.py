import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


''' PREPROCESSING DATA '''
# Load image data
image_mat = loadmat(file_name="images.mat", mat_dtype=True)

# Orient image to be upright
image_data_orig = np.array(image_mat['images']).T
for i in range(0, len(image_data_orig)):
    image_data_orig[i] = image_data_orig[i].T

# Flatten image data
image_data = image_data_orig.reshape(len(image_data_orig), -1)

# Normalize images' pixel values
image_data = image_data / 255.0


''' K-MEANS CLUSTERING '''
k = 5

# Random initialization
clusters = [0] * len(image_data)
for i in range(0, len(clusters)):
    clusters[i] = np.random.randint(0, k)

loss = [0,1]
kmeans = np.zeros((k, len(image_data[0])))

while(loss[-1] != loss[-2]):
    # Updating means and calculating overall loss
    error = 0
    for ki in range(0, k):
        # mean update for cluster ki
        inds = [i for i, j in enumerate(clusters) if j == ki]
        c = image_data[inds]
        kmeans[ki, :] = np.mean(c, 0)

        # error calculations for cluster ki
        for ci in range(0, len(c)):
            error += np.sum((c[ci] - kmeans[ki,:])**2)
            
    loss.append(error)
    print(error)

    # Updating cluster groupings
    for di in range(0, len(image_data)):
        dists = np.array([np.sum((image_data[di] - kj)**2) \
                 for kj in kmeans])
        clusters[di] = np.argmin(dists)
