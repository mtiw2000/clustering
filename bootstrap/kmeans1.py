# -*- coding: utf-8 -*-
"""
Created on Fri Mar 09 18:41:32 2018

@author: Manish
"""

import numpy as np

#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn; seaborn.set() # Plot styling

from sklearn.datasets import load_digits
from sklearn.datasets.samples_generator import make_blobs



X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)


plt.scatter(X[:, 0], X[:, 1], s=50);

kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

digits.data.shape
 
kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits.data)
kmeans.cluster_centers_.shape


fig, ax = plt.subplots(2, 5, figsize=(8, 3))
centers = kmeans.cluster_centers_.reshape(10, 8, 8)
for axi, center in zip(ax.flat, centers):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)
    

rand = np.random.RandomState(42)

X = rand.rand(10, 2)

plt.scatter(X[:,0],X[:,1],s=100)


a=X[:,np.newaxis,:]

b=X[np.newaxis,:,:]



differences = X[:,np.newaxis,:] - X[np.newaxis,:,:]

differences.shape

sq_differences = differences ** 2

dist_sq = sq_differences.sum(-1)

dist_sq.diagonal()



nearest = np.argsort(dist_sq,axis=1)


digits = load_digits()
digits.data.shape
