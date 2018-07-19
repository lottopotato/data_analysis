# main
import sys
import numpy as np

from plot import *
from numpy_process import *
from model_read import load
from run_sklearn import *
from run_tensorflow import *


class damageForm:
     def __init__(self):
          # 동기화 문제 있음.
          #self.T01 = ["1883485", "1872723"]
          #self.T07 = ["1710089", "1673293"]
          #self.T06 = [] all 마모
          self.T06 = []
          self.T01 = []
          self.T07 = []
     def returnList(self, name):
          if (name == "T01"):
               damageList = self.T01
          elif (name == "T06"):
               damageList = self.T06
          elif (name == "T07"):
               damageList = self.T07
          else:
               damageList = []
          return damageList

def main(argv):
     try:
          dataName = sys.argv[1]
          analysis_name = sys.argv[2]
     except IndexError:
          print(" data name list [ \"T01\" , \"T06\" , \"T07\" ] ")
          print(" basic \"original\"")
          print(" cluster [ \"kmeans\" , \"hierarchical\"")
          print(" unsupervised learning [\"autoEncoder\", \"GAN\"] ")
          print(" supervised learning [\"DNN\"]")
          print(" usage : main.py dataName clustering&unsupervised_Learning ")
          print(" or do input a field below.")
          dataName = input( " data name : ")
          analysis_name = input(" clustering & unsupervised learning name : ")
          
          
     data_name = [ "T01" , "T06" , "T07", "test" ]
     if not dataName in data_name:
          print(" data name list [ \"T01\" , \"T06\" , \"T07\" ] ")
          return False
     else:
          item, src = load(dataName, saveOp = False)

     src_arr = list2numpy(src)
     tick_arr = arrArange(src_arr.shape[1])
     print(" number of each id : " + str(src_arr.shape[0]))
     print(" ticks range of each id : " + str(src_arr.shape[1]))
     print(" ========================== \n")

     plot_setting(tick_arr, src_arr, item, dataName, analysis_name, save = True)

def plot_setting(tick_arr, src_arr, itemId, dataName, analysis_name, save):
     # damage label
     damageId = damageForm()
     damageList = damageId.returnList(dataName)
     # fig name init
     fig_name = str(dataName) + "_" + str(analysis_name)
     # basic
     if ( analysis_name == "original"):
          basic4info_fig(tick_arr, src_arr, itemId, fig_name, 0.1, save, damageList)
     else:
          # clustering. kmeans
          if( analysis_name == "kmeans"):
               compareName = "std"
               if not (compareName in ["mean", "var", "std"]):
                    print(" possible form : " + str(["mean", "var", "std"]))
                    return False
               n_cluster = 3
               fig_name += "_n_" + str(n_cluster) + "_"
               fig = kmeans(src_arr, n_cluster, compareName, itemId, 1, damageList)

          # clustering. Hierarchical-agglomerative
          elif( analysis_name == "hierarchical"):
               n_cluster = 3
               fig_name += "_n_" + str(n_cluster) + "_"
               fig = hgCluster(src_arr, n_cluster, "original", itemId, 1, damageList)
               
          # unsupervised-learning. auto-encoder
          elif( analysis_name == "autoEncoder"):
               fig, step =autoEncoder_run(src_arr, itemId, test =100, learning_rate = 0.01,
                                    step = 30, print_step = 5, damageList = damageList)
               fig_name += "_step_" + str(step) + "_"

          # unsupervised-learning. Generative Adversarial Network
          elif( analysis_name == "GAN"):
               fig, step = GAN_run(src_arr, itemId, test=10, learning_rate = 0.0003, step = 100, print_step = 10,
                       damageList = damageList)

               fig_name += "_step" + str(step) + "_"

          # supervised=learning. Deep Neural Network with Hierarchical - manhatten distance
          elif( analysis_name == "DNN"):
               fig, step = DNN_run(src_arr, itemId, test = 200, learning_rate = 0.001, step = 50, print_step = 10,
                                   damageList = damageList)
               fig_name += "_step" + str(step) + "_"

          # expand damage wave form test, * 100
          elif( analysis_name == "test"):
               dnn_test = DeepNeuralNet(src_arr, itemId, test = 10, learning_rate = 0.001, step = 1, print_step = 1)
               newArr, labels = dnn_test.getNewData()
               fig, plot = create_fig(1, 2)
               row_arr = arrArange(len(labels))
               plot[0].plot( row_arr, labels, linewidth = 0.1)
               damage = 0
               normal = 0
               for i in range(len(labels)):
                    if ( labels[i] == 1):
                         damage += 1
                         plot[1].plot(newArr[i])
                    else:
                         normal += 1
               print( "damage : %i normal : %i" %(damage, normal))
               
          else:
               print(" basic [\"original\"]")
               print(" cluster [ \"kmeans\" , \"hierarchical\"")
               print(" unsupervised learning [\"autoEncoder\", \"GAN\"] ")
               print(" supervised learning [\"DNN\"]")
               print( " error from main.py plot_setting")
               return False
               
          setting_fig(fig, "data analysis")
          plot_show(fig, fig_name, save)
          
if __name__ == "__main__":
     main(sys.argv)
