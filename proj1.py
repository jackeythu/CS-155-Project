# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 10:00:38 2016

@author: ruijiasun
"""

import numpy as np
def getData(name):
    data = np.genfromtxt(name,skip_header = 1, delimiter='|')
    y = data[:, len(data[0])-1]
    x = data[:, 0:len(data[0])-1]
    # to normalize the data
    #mean = np.mean(x, axis = 0)
    #std = np.std(x, axis = 0)
    #x = (x-mean)/std
    x = np.append(np.transpose(np.ones((1, len(x)))), x, axis = 1)
    for i in range(len(y)):
        if (y[i] == 0):
            y[i] = -1
    return x, y
    
def loss(x, y, w, lmd):
    sumloss = 0
    for j in range(len(x)):
        sumloss += (np.log(1+np.exp(-y[j]*np.dot(w,x[j,:]))) + lmd*np.dot(w,w)/len(x))
    return sumloss

def SGD(x, y, lmd):
    size = len(x);
    i = 0;
    epsilon = .0001
    eita = np.exp(-8)
    epoch = 1000
    #w = np.random.rand(len(x[0]))*0.1
    w = np.ones(len(x[0]))*0.1
    loss1 = 0.0
    diffloss = 0.0
    epochloss = []
    while (True):
        j = int(np.random.rand(1)*size%size)
        w -= eita*(-(y[j]*x[j,:]*np.exp(-y[j]*np.dot(w,x[j,:])))/(1+np.exp(-y[j]*np.dot(w, x[j,:]))) + 2*lmd/size*w)
        # the first loss of this epoch
        if (i == 0):
            loss1 = loss(x, y, w, lmd)
        # the last loss of this epoch
        elif (i == epoch):
            sumloss = loss(x, y, w, lmd)
            epochloss.append(sumloss)
            # diffloss is the loss reduction of the first epoch
            if (diffloss == 0.0):
                diffloss = sumloss - loss1;
            else:
                temp = sumloss - loss1;
                print abs(temp/diffloss)
                if (abs(temp/diffloss) < epsilon):
                    return w, epochloss[len(epochloss)-1]
            i = -1
        i = i+1

def predict(x, w):
    fo = open("pred.csv", "w")
    fo.write("Id,Prediction\n");
    for j in range(len(x)):
        if (1/(1+np.exp(-np.dot(w,x[j,:]))) > 0.5):
            fo.write(str(j+1)+", 1\n")
        else:
            fo.write(str(j+1)+", 0\n");
    fo.close()
         
x,y = getData('training_data.txt');
w1, unitloss1 = SGD(x, y, 0.1)

xtest = np.genfromtxt("testing_data.txt",skip_header = 1, delimiter='|');
xtest = np.append(np.transpose(np.ones((1, len(xtest)))), xtest, axis = 1)
predict(xtest, w1)