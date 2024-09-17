import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv('Mall_Customers.csv')

df = df.drop('Gender',axis=1)

X = df[['Annual Income (k$)','Spending Score (1-100)']].values

wcss = []

for i in range(1,11):
    km = KMeans(n_clusters=i)
    km.fit(df)
    wcss.append(km.inertia_)

kmeans = KMeans(n_clusters=2)
y_kmeans = kmeans.fit_predict(X)

colors = ['red','blue']

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], c='red', label='Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], c='blue', label='Cluster 2')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='yellow', marker='X', label='Centroids')

plt.show()