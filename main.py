# main

from plot import *
from numpy_process import *
from model_read import load
from run_sklearn import *
from data_type import dataType

dataType = dataType()

def main():
     x, y = load()

     x_arr = list2numpy(x)
     y_arr = list2numpy(y)

def plot_setting(x_arr, y_arr):
     kmeans_x, kmeans_y = division_pairArr( kmeans_each_sample(x_arr, y_arr) )

     fig, plot = create_fig(1,2)
     scatter(plot[0], x_arr, y_arr, "original", "blue", 0.1)
     scatter(plot[1], kmeans_x, kmeans_y, "k-means k=10", "red", 0.1)
     setting_fig(fig, "data analysis")
     plot_show()

def kmeans_each_sample(x_arr, y_arr):
     total_len = x_arr.shape[0]
     each_len = int( total_len / dataType.SRC_len )
     resultList = []
     for i in range(each_len):
          x, y = sampling_arr(x_arr, y_arr, i)
          merge_arr = merge_pairArr(x, y)
          kmeans_arr = kmeans(merge_arr, 10)
          resultList.extend(kmeans_arr.tolist())
          print( " k-means processing - {} ".format(i) + " / " + " {} ".format(each_len), end="\r")

     print("\n k-means processing done ", end="\n")
     result_arr = list2numpy(resultList)
     return result_arr



if __name__ == "__main__":
     main()
