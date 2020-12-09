#!/usr/bin/env python
# coding: utf-8

# # THE SPARK FOUNDATION

# ### Prediction Using Unsupervised Learning

# #### Predict the optimum number of clusters and visual representation

# In[9]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans

iris = pd.read_csv('Iris.csv')
# get all data
iris.head(-1) 


# In[10]:


x = iris.iloc[:, [0, 1, 2, 3]].values

lst = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    lst.append(kmeans.inertia_)
    
plt.figure(figsize=(8,6))
plt.plot(range(1, 11), lst)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Within cluster sum of squares') 
plt.show()


# In[14]:


km = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)
y_km = km.fit_predict(x)
plt.figure(figsize=(9,6))
plt.scatter(x[y_km == 0, 0], x[y_km == 0, 1], 
            s = 100, c = 'red', label = 'IRIS-SETOSA')
plt.scatter(x[y_km == 1, 0], x[y_km == 1, 1], 
            s = 100, c = 'blue', label = 'IRIS-VERSICOLOUR')
plt.scatter(x[y_km == 2, 0], x[y_km == 2, 1],
            s = 100, c = 'green', label = 'IRIS-VIRGINICA')

plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'CENTROID')

plt.legend()


# #### FINAL CLUSTERING
