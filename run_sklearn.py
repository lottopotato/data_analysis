# clustering
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import numpy as np

from numpy_process import *
from data_type import dataType

def kmeans(arr, n_cluster):
     kmeans = KMeans(n_clusters = n_cluster).fit(arr)
     y = kmeans.cluster_centers_
     return y

def printOption(n_cluster, total_len, row_len):
     print(" ========================== ")
     print(" K-means cluster ")
     print(" number of cluster : " + str(n_cluster))
     print(" data total length : " + str(total_len))
     print(" data row each item id  : " + str(row_len))
     print(" ========================== ")

def kmeans_each_sample(x_arr, y_arr, numberOfCluster):
     dataAt = dataType()
     total_len = x_arr.shape[0]
     each_len = int( total_len / dataAt.SRC_len )
     resultList = []
     printOption(numberOfCluster, total_len, each_len)
     for i in range(each_len):
          x, y = sampling_arr(x_arr, y_arr, i)
          merge_arr = merge_pairArr(x, y)
          kmeans_arr = kmeans(merge_arr, numberOfCluster)
          resultList.extend(kmeans_arr.tolist())
          print( " k-means processing - {} ".format(i) + " / " + " {} ".format(each_len), end="\r")

     print("\n k-means processing done ", end="\n")
     result_arr = list2numpy(resultList)
     return result_arr

