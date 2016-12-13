import numpy as np
import csv
from sklearn.feature_extraction import DictVectorizer

''' Featurizing TRAIN Data '''
train_file = np.array(list(csv.DictReader(open("train_data.csv"))))

feat_keys = train_file[0].keys()
featval_counts = dict.fromkeys(feat_keys)
for feat in featval_counts:
    featval_counts[feat] = {}
    
for row in train_file:
    for feat in row:
        featval = row[feat]
        if featval.isdigit(): continue
        if featval in featval_counts[feat]:
            featval_counts[feat][featval] += 1
        else:
            featval_counts[feat][featval] = 1

featval_modes = dict.fromkeys(feat_keys)
for feat in featval_counts:
    maxcount = 0
    for val in featval_counts[feat]:
        if featval_counts[feat][val] > maxcount:
            featval_modes[feat] = val
            maxcount = featval_counts[feat][val]
            

for row in train_file:
    for item in row:
        if row[item].isdigit():
            row[item] = float(row[item])
        if row[item] == '?':
            row[item] = featval_modes[item]

train_vec = DictVectorizer()

traindv = train_vec.fit_transform(train_file)

train_data = traindv.toarray()

feat_names = train_vec.get_feature_names()
label_ind = feat_names.index('label')

train_labels = train_data[:, label_ind]
train_data = np.delete(train_data, label_ind, 1)


''' Featurizing TEST Data '''
test_file = np.array(list(csv.DictReader(open("test_data.csv"))))

feat_keys = test_file[0].keys()
featval_counts = dict.fromkeys(feat_keys)
for feat in featval_counts:
    featval_counts[feat] = {}
    
for row in test_file:
    for feat in row:
        featval = row[feat]
        if featval.isdigit(): continue
        if featval in featval_counts[feat]:
            featval_counts[feat][featval] += 1
        else:
            featval_counts[feat][featval] = 1

featval_modes = dict.fromkeys(feat_keys)
for feat in featval_counts:
    maxcount = 0
    for val in featval_counts[feat]:
        if featval_counts[feat][val] > maxcount:
            featval_modes[feat] = val
            maxcount = featval_counts[feat][val]
            

for row in test_file:
    for item in row:
        if row[item].isdigit():
            row[item] = float(row[item])
        if row[item] == '?':
            row[item] = featval_modes[item]

testdv = train_vec.transform(test_file)

test_data = testdv.toarray()
test_data = np.delete(test_data, label_ind, 1)
