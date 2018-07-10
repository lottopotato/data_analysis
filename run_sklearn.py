from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import numpy as np

def kmeans(arr, n_cluster):
     kmeans = KMeans(n_clusters = n_cluster).fit(arr)
     y = kmeans.cluster_centers_
     return y


