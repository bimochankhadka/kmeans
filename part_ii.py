'''
   @importing the libaries
   @pandas
   @matplotlib
   @scikit-learn
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Orange.data import Table, Domain
from sklearn.cluster import KMeans


#Read csv
data_tab = pd.read_csv('ionoshpere.csv')


v=data_tab.drop(['y'],axis=1)
X=v.values


#kmeans model with cluster of 2 
k_means = KMeans(n_clusters = 2,random_state = 0).fit(X)


#prediction
y_kmeans = k_means.predict(X)


#centroid calculation
centroids = k_means.cluster_centers
labels= k_means.labels

#plotting the clusters
plt.plot(X[labels==0,0],X[labels==0,1],'r.', label='cluster 1')
plt.plot(X[labels==1,0],X[labels==1,1],'b.', label='cluster 2')
plt.plot(centroids[:,0],centroids[:,1],'mo',markersize=8, label='centroids')
plt.legend(loc='best')
plt.show()


kmeans_model = KMeans(n_clusters = 2, random_state = 0).fit(X)
print(kmeans_model.labels_.data)
print(dir(kmeans_model.labels))
