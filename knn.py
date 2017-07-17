# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 01:41:40 2017

@author: Sid007
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import neighbors , preprocessing
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap


hx1=np.random.normal(0, 10, 10000)
hx2=np.random.normal(0, 1, 10000)
hy1=np.zeros(10000)

px1=np.random.normal(30, 10, 10000)
px2=np.random.normal(4, 1, 10000)
py1=np.ones(10000)

list = np.zeros((20000,3),float)
list_nml = np.zeros((20000,3),float)

x1_val = np.concatenate((hx1,px1))
x2_val = np.concatenate((hx2,px2))

x1_diff = max(x1_val)-min(x1_val)
x2_diff = max(x2_val)-min(x2_val)

x1_normalized = [x1/(x1_diff) for x in x1_val]
x2_normalized = [x2/(x2_diff) for y in x2_val]
xy_normalized = np.vstack([x1_normalized,x2_normalized]).T

X1 = np.append(hx1,px1)
X2 = np.append(hx2,px2)
X1 = X1.reshape(20000,1)
X2 = X2.reshape(20000,1)

nml = preprocessing.MinMaxScaler()
X1_nml = nml.fit_transform(X1)
X2_nml = nml.fit_transform(X2)

hx1_nml = X1_nml[10000:]
px1_nml = X1_nml[:10000]
hx2_nml = X2_nml[10000:]
px2_nml = X2_nml[:10000]

for i in range (0,10000):
    list[i][0] = hx1[i]
    list[i][1] = hx2[i]
    list[i][2] = hy1[i]
    list_nml[i][0] = hx1_nml[i]
    list_nml[i][1] = hx2_nml[i]
    list_nml[i][2] = hy1[i]
    i+1

for y in range(0, 10000):
    list[y+10000][0] = px1[y]
    list[y+10000][1] = px2[y]
    list[y+10000][2] = py1[y]
    list_nml[y+10000][0] = px1_nml[y]
    list_nml[y+10000][1] = px2_nml[y]
    list_nml[y+10000][2] = py1[y]
    y+1

random.shuffle(list)
train_data = list[16000:]
test_data = list[:4000]

X = train_data[:,:2]
y = train_data[:,2]

X_test = test_data[:,:2]
y_test = test_data[:,2]

knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(X,y)
pred = knn.predict(X_test)

random.shuffle(list_nml)
train_data_nml = list_nml[16000:]
test_data_nml = list_nml[:4000]

X_nml = train_data_nml[:,:2]
y_nml = train_data_nml[:,2]

X_test_nml = test_data_nml[:,:2]
y_test_nml = test_data_nml[:,2]

knn.fit(X_nml,y_nml)
pred_nml = knn.predict(X_test_nml)



plt.scatter(hx1,hx2,s=10, alpha=0.5)
plt.scatter(px1,px2,s=10, alpha=0.5)
plt.show()
print ("error rate: "+str(1-accuracy_score(pred,y_test))+" %")


plt.scatter(hx1_nml,hx2_nml,s=10, alpha=0.5)
plt.scatter(px1_nml,px2_nml,s=10, alpha=0.5)
plt.show()
print ("error rate: "+str(1-accuracy_score(pred_nml,y_test_nml))+" %")