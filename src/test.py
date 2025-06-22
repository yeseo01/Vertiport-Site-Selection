import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv('/Users/yeseo/Desktop/항공우주AI기초/Project_1/project1_3/VertiportProcessed_Final.csv')
X = data[['Latitude', 'Longitude']]

kmeans = KMeans(n_clusters=17)
kmeans.fit(X)
labels = kmeans.predict(X)
centroids = kmeans.cluster_centers_

plt.scatter(X['Latitude'], X['Longitude'], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.75)
plt.title('K-Means Clustering')
plt.ylabel('Latitude')
plt.xlabel('Longitude')
plt.show()


