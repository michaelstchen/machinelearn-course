import numpy as np
import DecisionTree as dt
from scipy.io import loadmat

def benchmark(pred_labels, true_labels):
    errors = pred_labels != true_labels
    err_rate = sum(errors) / float(len(true_labels))
    indices = errors.nonzero()
    return err_rate, indices

execfile('featurize.py')

#spam_data = loadmat(file_name="spam_data32.mat", mat_dtype=True)
spam_data = loadmat(file_name="spam_data.mat", mat_dtype=True)
train_data = np.array(spam_data['training_data'])
train_labels = np.transpose(np.array(spam_data['training_labels']))
train_labels = train_labels[:, 0]

#train_data = train_data[:3000]
#train_labels = train_labels[:3000]

random_state = np.random.get_state()
np.random.shuffle(train_data)
np.random.set_state(random_state)
np.random.shuffle(train_labels)

k = len(train_data) * 2 / 3
valid_data = train_data[k:]
valid_labels = train_labels[k:]
train_data = train_data[0:k]
train_labels = train_labels[0:k]

dectree = dt.DecisionTree(10, len(train_data[0]), 10)
dectree.train(train_data, train_labels)

pred_labels = dectree.predict(valid_data)
err, ind = benchmark(pred_labels, valid_labels)
