# clustering
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering as aggClusters
import time

#import matplotlib.pyplot as plt
#import numpy as np

from numpy_process import *
from plot import *

def printOption(cluster_name, n_cluster, total_len, input_dataName, damageList):
     print(" ========================== ")
     print(" cluster : " + cluster_name)
     print(" number of cluster : " + str(n_cluster))
     print(" use data : " + str(input_dataName))
     print(" data total row length : " + str(total_len))
     print(" damaged item id : " + str(damageList))
     print(" ========================== ")

# class kmean
def kmeans_run(arr, n_cluster):
     start = time.time()
     kmeans = KMeans(n_clusters = n_cluster).fit(expandDims(arr,1))
     x = kmeans.cluster_centers_
     y = kmeans.labels_
     final = time.time()
     print("\n Clustering complete.  time : {}\n".format(str(final - start)))
     return x.reshape(n_cluster), y

def kmeans(src_arr, numberOfCluster, input_dataName, itemId, thickness, damageList):
     input_data = array_cal_each_id(src_arr, cal= input_dataName)

     result_arr = np.zeros([src_arr.shape[0], numberOfCluster])
     printOption("kmeans", numberOfCluster, src_arr.shape[0], input_dataName, damageList)
     
     center, label = kmeans_run(input_data, numberOfCluster)
     return kmeans_plot(src_arr, label, src_arr.shape[0], input_data, input_dataName, itemId, thickness, damageList)
     
def kmeans_plot(src_data, label, data_row, input_data, input_dataName, itemId, thickness, damageList):
     fig, plot = create_fig(2,2)
     plot[1,0].set_visible(False)
     plot[1,1].set_visible(False)
     linePlot = addPlot_under(fig, 2)
     scatter(plot[0,0], arrArange(data_row), input_data, None, input_dataName, "blue", thickness)
     for i in range(data_row):
          if(label[i] == 0):
               setColor = "blue"
          elif(label[i] == 1):
               setColor = "yellow"
          elif(label[i] == 2):
               setColor = "green"
          else:
               print(" error kmeans")
               return False
          print(" - drawing plot ... {}".format(i+1) + " / " + "{}".format(data_row) , end = "\r")
          scatter(plot[0,1], i, input_data[i], None, "K-means clustering", setColor, thickness)
          line(linePlot, src_data[i], None, "original", "original", setColor, thickness, option="singleArr")
          if( str(itemId[i]) in damageList):
               line(linePlot, src_data[i], None, "damage", "original", "red", 3, option="singleArr")
               scatter(plot[0,0], i, input_data[i], "damage", input_dataName, "red", 5)
               scatter(plot[0,1], i, input_data[i], "damage", input_dataName, "red", 5)

     #legend
     legend_list = []
     for color, label in zip(["blue", "yellow", "green", "red"], ["label 1", "label 2", "label 3", "damage"]):
          legend_list.append(legend_label(color, label))
     
     plot_legend(plot[0,1], legend_list)
     plot_legend(linePlot, legend_list)
     print("\n")
     return fig
          
# class Hierarchical-agglomerative-clustering
def hgCluster_run(arr, n_cluster):
     start = time.time()
     modelList = []
     for index, metric in enumerate(["cosine", "euclidean", "cityblock"]):
          #arr = np.expand_dims(arr, axis=0)
          model = aggClusters(n_clusters=n_cluster, linkage="average", affinity=metric).fit(arr)
          modelList.append(model.labels_)
     final = time.time()
     print("\n Clustering complete.  time : {}\n".format(str(final - start)))
     return modelList

def hgCluster_single_metric(arr, n_cluster):
     start = time.time()
     model = aggClusters(n_clusters = n_cluster, linkage = "average", affinity = "cityblock")
     model.fit(arr)
     final = time.time()
     print("\n Clustering complete.  time : {}\n".format(str(final - start)))
     return model.labels_
               
def hgCluster(src_data, numberOfCluster, input_dataName, itemId, thickness, damageList):
     printOption("Hierarchical-agglomerative", numberOfCluster, src_data.shape[0], input_dataName, damageList)
     hgClusterList = hgCluster_run(src_data, numberOfCluster)
     
     fig, plot = create_fig(2,2)
     for i in range(src_data.shape[0]):
          if( str(itemId[i]) in damageList):
               line(plot[0,0], arrArange(src_data.shape[1]), src_data[i], None, "original", "red", 2)
          else:
               line(plot[0,0], arrArange(src_data.shape[1]), src_data[i], None, "original", "blue", thickness)
          print(" - drawing plot ... {}".format(i+1) + " / " + "{}".format(src_data.shape[0]) , end = "\r")
     print("\n")
     legend_list = []
     for label, color, label_name in zip(arrArange(numberOfCluster), "byg", ["label 1", "label 2", "label 3"]):
          legend_list.append(legend_label(color, label_name))
          temp = src_data[hgClusterList[0] == label].T
          line(plot[0,1], temp, None, "cosine", "cosine", color, thickness, option="singleArr")
          temp2 = src_data[hgClusterList[1] == label].T
          line(plot[1,0], temp2, None, "euclidean", "euclidean", color, thickness, option="singleArr")
          temp3 = src_data[hgClusterList[2] == label].T
          line(plot[1,1], temp3, None, "cityblock", "cityblock", color, thickness, option="singleArr")
          print(" - drawing plot ... {}".format(label+1) + " / " + "{}".format(numberOfCluster) , end = "\r")
     print("\n")
     legend_list.append(legend_label("red", "damage"))
     plot_legend(plot[0,1], legend_list)
     plot_legend(plot[1,0], legend_list)
     plot_legend(plot[1,1], legend_list)
                       
     return fig

def checking(labels, itemId, check):
     for i in range(len(labels)):
          for j in range(3):
               if (labels[i] == check):
                    print(itemId[i])
