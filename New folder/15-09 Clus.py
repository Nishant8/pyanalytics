# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 20:53:29 2020

@author: Nishant Agarwal
"""

#Topic:Clustering - marks and mtcars
#-----------------------------
#libraries

import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

data = {'x': [25,34,22,27,33,33,31, 22,35,34,67,54,57,43,50,57,59,52,65, 47,49,48,35,33,44,45,38,43,51,46],'y': [79,51,53,78,59,74,73,57,69,75,51,32, 40,47,53,36,35,58, 59,50,25,20,14,12,20,5,29,27,8,7]       }
  
df = pd.DataFrame(data,columns=['x','y'])
print (df)

plt.scatter(df['x'],df['y'])



kmeans = KMeans(n_clusters=).fit(df)
centroids = kmeans.cluster_centers_
print(centroids)

plt.scatter(df['x'], df['y'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.show()

#%% 4 clusters
kmeans = KMeans(n_clusters=4).fit(df)
centroids = kmeans.cluster_centers_
print(centroids)

plt.scatter(df['x'], df['y'], c= kmeans.labels_.astype(float), s=50, alpha=0.8)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.show()


#mtcars

from pydataset import data
mtcars = data('mtcars')
data = mtcars.copy()
id(data)
data
 
kmeans = KMeans( init = 'random', n_clusters=2, max_iter=300)
kmeans
kmeans.fit(data)

kmeans.cluster_centers_ 



#need for scaling : height & weight are in different scales
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data)
scaled_features[:5]  #values between -3 to +3

kmeans = KMeans( init = 'random', n_clusters=2, max_iter=300)
kmeans
kmeans.fit(scaled_features)
kmeans.inertia_
kmeans.cluster_centers_  #average or rep values
kmeans.n_iter_  #in 6 times, clusters stabilised
kmeans.labels_
kmeans.cluster_centers_.shape
kmeans.cluster_centers_[0:1]
#https://realpython.com/k-means-clustering-python/




data1=data
data1["labels"]=kmeans.labels_
data1




#Topic: Clustering
#-----------------------------

#libraries
import matplotlib.pyplot as plt
#pip install kneed
from kneed import KneeLocator
from sklearn.datasets import make_blobs  #simulate data
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#parameters for creating simulated data
#n_samples - total samples to generate
#centers - no of centers or clusters
#cluster_std - standard deviation

#simulated / synthentic data
features, true_labels = make_blobs( n_samples=200, centers=3, cluster_std=2.75, random_state=42)
features[:5]
features.shape
true_labels[:5]
true_labels.shape
#need for scaling : height & weight are in different scales
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
scaled_features[:5]  #values between -3 to +3

kmeans = KMeans( init = 'random', n_clusters=3, n_init=10, max_iter=300, random_state=42)
kmeans
kmeans.fit(scaled_features)
kmeans.inertia_
kmeans.cluster_centers_  #average or rep values
kmeans.n_iter_  #in 6 times, clusters stabilised
kmeans.labels_[:5]


#%%choosing no of clusters
kmeans_kwargs = {'init':'random', 'n_init':10, 'max_iter': 300, 'random_state': 42,}
sse=[]
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_features)
    sse.append(kmeans.inertia_)

plt.style.use('fivethirtyeight')
plt.plot(range(1,11), sse)
plt.xticks(range(1,11))
plt.xlabel('No of clusters')
plt.ylabel('SSE')
plt.show();

kl = KneeLocator(x=range(1,11), y=sse, curve='convex', direction='decreasing')
kl.elbow


##https://realpython.com/k-means-clustering-python/