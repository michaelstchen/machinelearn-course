import numpy as np
import math
import csv
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import metrics, svm

#benchmark.m, converted
def benchmark(pred_labels, true_labels):
    errors = pred_labels != true_labels
    err_rate = sum(errors) / float(len(true_labels))
    indices = errors.nonzero()
    return err_rate, indices

train_set = loadmat(file_name="train.mat")
train_data = np.transpose(train_set["train_images"])
train_labels = train_set["train_labels"].reshape(1,-1)[0]

numImages = train_data.shape[0]
height = train_data.shape[1] 
width = train_data.shape[2]

train_data_flat = train_data.reshape(numImages, -1)

random_state = np.random.get_state()
np.random.shuffle(train_data)
np.random.set_state(random_state)
np.random.shuffle(train_labels)

train_nums = [100, 200, 500, 1000, 2000, 5000, 10000]
valid_errs = [0, 0, 0, 0, 0, 0, 0]
pred_labels_array = np.zeros((len(train_nums), 10000));
for i in range(0,len(train_nums)):
    print(train_nums[i])
    clf = svm.SVC(kernel='linear')
    clf.fit(train_data_flat[10000:10000+train_nums[i]], \
            train_labels[10000:10000+train_nums[i]])
    pred_labels = clf.predict(train_data_flat[:10000])
    err_rate, indices = benchmark(pred_labels, train_labels[:10000])
    valid_errs[i] = 1-err_rate
    pred_labels_array[i,:] = np.array(pred_labels)
    
with open('prob1_validation.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(['# of Images'] + ['% Correct'])
    for i in range(0, len(train_nums)):
        writer.writerow([train_nums[i]] + [valid_errs[i]])


        
plt.plot(train_nums, [1 - x for x in valid_errs])
plt.title('Effect of Number of Training Samples on Error Rate')
plt.tight_layout()
plt.xlabel('Number of Training Images')
plt.ylabel('Error Rate')
plt.savefig('err_plot')
plt.close()
