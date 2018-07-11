# main
import sys

from plot import *
from numpy_process import *
from model_read import load
from run_sklearn import *
from run_tensorflow import *

from data_type import dataType

dataType = dataType()

def main(argv):
     try:
          dataName = sys.argv[1]
          name = sys.argv[2]
     except IndexError:
          print(" data name list [ \"T01\" , \"T06\" , \"T07\" ] ")
          print(" clustering & unsupervised learning list [ \"original\" , \"kmeans\" , \"autoEncoder\" ] ")
          print(" usage : main.py dataName clustering&unsupervised_Learning ")
          print(" or do input a field below.")
          dataName = input( " data name : ")
          name = input(" clustering & unsupervised learning name : ")
          
          
     data_name = [ "T01" , "T06" , "T07" ]
     if not dataName in data_name:
          print(" data name list [ \"T01\" , \"T06\" , \"T07\" ] ")
          return False
     else:
          x, y = load(dataName, saveOp = False)

     x_arr = list2numpy(x)
     y_arr = list2numpy(y)

     # clustering & unsupervised learning [ "original" , "kmeans" , "autoEncoder" ]
     plot_setting(x_arr, y_arr, name)

def plot_setting(x_arr, y_arr, name):
     if ( name == "original"):
          basic4info_fig(x_arr, y_arr)
     else:
          # kmeans
          if( name == "kmeans"):
               cluster_Arr = kmeans_each_sample(x_arr, y_arr, 10)
               new_x, new_y = division_pairArr(cluster_Arr)
          # autoEncoder
          elif( name == "autoEncoder"):
               x_arr, y_arr, newY_arr = autoEncoder_run(x_arr, y_arr, test = 100, learning_rate = 0.01,
                                                        step = 100, print_step = 10)
               new_x = x_arr
               new_y = reshape_2dto1d(newY_arr)
          else:
               print(" clustering & unsupervised learning list [ \"original\" , \"kmeans\" , \"autoEncoder\" ] ")
               print( " error from main.py plot_setting")
               return False
          
          fig, plot = create_fig(1,2)
          scatter(plot[0], x_arr, y_arr, "original", "blue", 0.1)
          scatter(plot[1], new_x, new_y, name, "red", 0.1)
          setting_fig(fig, "data analysis")
          plot_show(fig, name, save = True)
          

if __name__ == "__main__":
     main(sys.argv)
