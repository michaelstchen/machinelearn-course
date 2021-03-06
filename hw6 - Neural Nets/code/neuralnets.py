# My Neural Nets Class and Associated Logic

import numpy as np
import pdb

def benchmark(pred_labels, true_labels):
    errors = pred_labels != true_labels
    err_rate = sum(errors) / float(len(true_labels))
    indices = errors.nonzero()
    return err_rate, indices

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh(z):
    return np.tanh(z)

class NeuralNet:
    # N is number input, M is num hidden, O is num output
    # LOSSFN is 0 for MSE and 1 for Cross-Entropy
    def __init__(self, n, h, o, lossfn):
        self.n = n
        self.h = h
        self.o = o
        self.lossfn = lossfn

    # W1 is weight vector for input-hidden
    # W2 is weight vector for hidden-output
    def initialize_weights(self):
        self.W1 = np.random.normal(0, 0.01, (self.h, self.n+1))
        self.W2 = np.random.normal(0, 0.01, (self.o, self.h+1))

    # HID is the vector of hidden unit values
    # Y is the vector of output values
    def forward(self, X):
        self.hid = tanh(np.dot(self.W1, X))
        self.hid = np.r_[self.hid, 1]
        self.y = sigmoid(np.dot(self.W2, self.hid))

    # Compute gradients for the weights
    def backprop(self, X, yhat):
        if (self.lossfn == 0):
            delta2 = np.multiply(-(yhat-self.y), self.sigmoidprime())
        elif (self.lossfn == 1):
            delta2 = self.y - yhat
        dJdW2 = np.outer(delta2, self.hid)

        delta1 = np.multiply(np.dot(self.W2.T,delta2),self.tanhprime())
        dJdW1 = np.outer(delta1, X)

        return dJdW1, dJdW2

    def trainNeuralNet(self, samples, labels, step, epochs):
        loss_list = []
        acc_list = []
        for i in range(1, epochs+1):
            print("Epoch: " + str(i))
            for j in range(0, len(samples)):
                if (j % 1000 == 0):                    
                    print("  samples - " + str(j))
                    if (self.lossfn == 0):
                        loss, acc = self.meansqerr(samples, labels)
                    elif (self.lossfn == 1):
                        loss, acc = self.crossenterr(samples, labels)
                    loss_list.append(loss)
                    acc_list.append(acc)
                    
                r = np.random.randint(0, len(samples))
                X = np.r_[samples[r], 1]
                yhat = labels[r]
                
                self.forward(X)
                dJdW1, dJdW2 = self.backprop(X, yhat)

                self.W1 = self.W1 - step * dJdW1[:-1]
                self.W2 = self.W2 - step * dJdW2

        return loss_list, acc_list

    def predictNeuralNet(self, samples):
        labels = np.zeros((len(samples), self.o))
        for i in range(0, len(samples)):
            X = np.r_[samples[i], 1]
            self.forward(X)
            labels[i] = (self.y).T
        return labels

    def sigmoidprime(self):
        return np.multiply(self.y, (1 - self.y))

    def tanhprime(self):
        return 1 - self.hid**2

    def meansqerr(self, samples, labels):
        preds = self.predictNeuralNet(samples)
        loss = np.sum((preds - labels)**2) / 2.0

        acc, ind = benchmark(np.argmax(preds, 1), \
                             np.argmax(labels, 1))

        return loss, 1-acc

    def crossenterr(self, samples, labels):
        preds = self.predictNeuralNet(samples)

        left = np.multiply(labels, np.log(preds+10e-99))
        right = np.multiply(1-labels, np.log(1-preds+10e-99))
        loss = -np.sum(left + right)

        acc, ind = benchmark(np.argmax(preds, 1), \
                             np.argmax(labels, 1))

        return loss, 1-acc

