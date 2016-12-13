import numpy as np
import random

class Node:
    def __init__(self, f, left, right, label):
        self.split_rule = f
        self.left = left
        self.right = right
        self.label = label


class DecisionTree:
    def __init__(self, depth, m, n):
        self.d = depth
        self.m = m
        self.n = n
        self.root = None

    def segmenter(self, data, labels):
        numsamps = len(data)
        #numfeats = len(data[0])
        feats = range(0, len(data[0]))
        random.shuffle(feats)

        best = (None, None)
        best_entropy = float("inf")
        #for f in range(0, numfeats):
        for f in feats[:self.m]:
            all_feat = data[:, f]
            splits = find_splits(np.unique(all_feat))
            for s in splits:
                left_hist = np.zeros(2)
                right_hist = np.zeros(2)
                for i in range(0, numsamps):
                    if all_feat[i] < s:
                        left_hist[labels[i]] += 1
                    else:
                        right_hist[labels[i]] += 1
            
                curr_entropy = impurity(left_hist, right_hist)
                if (curr_entropy < best_entropy):
                    best_entropy = curr_entropy
                    best = (f, s)

        return best, best_entropy

        
    def growTree(self, data, labels, depth):
        numData = len(labels)
        numOnes = np.count_nonzero(labels)
        numZeros = numData - numOnes
        if(depth==0 or numOnes==0 or numZeros==0 or numData<self.n):
            if (numOnes > numZeros):
                return Node(None,None,None,1)
            else:
                return Node(None,None,None,0)
        else:
            split, entropy = self.segmenter(data,labels)
            #This occurs when all data have same features
            if (entropy == float("inf")):
                if (numOnes > numZeros):
                    return Node(None,None,None,1)
                else:
                    return Node(None,None,None,0)
            
            l = np.where(data[:, split[0]] < split[1])
            r = np.where(data[:, split[0]] >= split[1])
            return Node(split, \
                        self.growTree(data[l],labels[l],depth-1), \
                        self.growTree(data[r],labels[r],depth-1), \
                        None)

        
    def train(self, data, labels):
        split, entropy = self.segmenter(data, labels)
        l = np.where(data[:, split[0]] < split[1])
        r = np.where(data[:, split[0]] >= split[1])
        self.root = Node(split, \
                         self.growTree(data[l],labels[l],self.d), \
                         self.growTree(data[r],labels[r],self.d), \
                         None)

    def predict(self, data):
        p = np.zeros(len(data))
        for i in range(0, len(data)):
            currNode = self.root
            while (currNode.label == None):
                f = currNode.split_rule[0]
                val = currNode.split_rule[1]
                if (data[i][f] < val):
                    currNode = currNode.left
                else:
                    currNode = currNode.right
            p[i] = currNode.label
        return p
        

def entropy(hist, count):
    p = hist / count
    surprise = -np.multiply(p, np.log2(p + 10e-10))
    return np.sum(surprise)

def impurity(left_hist, right_hist):
    S_l = np.sum(left_hist)
    S_r = np.sum(right_hist)
    weighted_Hl = S_l * entropy(left_hist,S_l)
    weighted_Hr = S_r * entropy(right_hist, S_r)
    return (weighted_Hl + weighted_Hr) / (S_l + S_r)

def find_splits(dataset):
    splits = (dataset + np.roll(dataset, 1)) / 2.0
    return splits[1::max(len(splits)//4, 1)]


