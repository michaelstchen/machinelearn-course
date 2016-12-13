import numpy as np
import math
import DecisionTree as dt
from random import randint
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

train_data = train_data[:3000]
train_labels = train_labels[:3000]

random_state = np.random.get_state()
np.random.shuffle(train_data)
np.random.set_state(random_state)
np.random.shuffle(train_labels)

ntrain = len(train_data) * 2 / 3
dfeat = len(train_data[0])

valid_data = train_data[ntrain:]
valid_labels = train_labels[ntrain:]
train_data = train_data[0:ntrain]
train_labels = train_labels[0:ntrain]

print("Generating Random Forest...")
T = 30
tree_list = []
for t in range(0, T):
    print("T = " + str(t))
    rind = [randint(0, ntrain-1) for x in range(0, ntrain)]
    dectree = dt.DecisionTree(50, int(math.sqrt(dfeat))+1, 10)
    dectree.train(train_data[rind], train_labels[rind])
    tree_list.append(dectree)

print("Predicting...")
pred_labels = np.zeros(len(valid_labels))
for dt in tree_list:
    pred_labels += dt.predict(valid_data)

pred_labels = np.round(pred_labels / float(T))


err, ind = benchmark(pred_labels, valid_labels)
print("error rate: " + str(err))

