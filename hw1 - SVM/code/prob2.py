import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

try:
    train_nums
    pred_labels_array
except NameError:
    execfile('prob1.py')

def write_cm_image(cm, num_im):
    plt.imshow(cm, interpolation='nearest')
    plt.title('# of Training Images: ' + str(num_im))
    plt.tight_layout()
    plt.colorbar()
    plt.xlabel('True label')
    plt.ylabel('Predicted label')
    plt.savefig(str(num_im), bbox_inches='tight')
    plt.close()

for i in range(0, len(train_nums)):
    cm = metrics.confusion_matrix(train_labels[:10000], pred_labels_array[i,:])
    write_cm_image(cm, train_nums[i])
    
