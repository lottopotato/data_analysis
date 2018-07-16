import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as date

import os, datetime

from numpy_process import *

def create_fig(dimension_x, dimension_y):
     if (dimension_x == None or dimension_y == None):
          flg, plot = plt.subplots()
     else:
          flg, plot = plt.subplots(dimension_x, dimension_y)
     return flg, plot

def setting_fig(fig, title):
     fig.canvas.set_window_title(title)

def addPlot_under(fig, location):
     addPlot = fig.add_subplot(location,1,location)
     return addPlot

def scatter(plot, x_data, y_data, name, plot_name, setColor, scale, option = "add"):
     if (option == "basic"):
          plt.scatter(x_data, y_data, color = setColor, s = scale, label = name)
          plt.title(plot_name)
     else:
          plot.scatter(x_data, y_data, color = setColor, s= scale, label = name)
          plot.set_title(plot_name)

def line(plot, x_data, y_data, name, plot_name, setColor, thickness, option = "add"):
     if (option == "basic"):
          plt.plot(x_data, y_data, color = setColor, linewidth = thickness, label = name)
          plt.title(plot_name)
     elif(option == "singleArr"):
          plot.plot(x_data, color = setColor, linewidth = thickness, label = name)
          plot.set_title(plot_name)
     else:
          plot.plot(x_data, y_data, color = setColor, linewidth = thickness, label = name)
          plot.set_title(plot_name)

def basic4info_fig(x_arr, y_arr, itemId, fig_name, thickness, save = True, damageList = []):
     print(" ========================== ")
     print(" basic original plot ")
     print(" original / mean / var / std ")
     print(" damaged imem id : " + str(damageList))
     print(" ========================== ")

     data_row = y_arr.shape[0]
     basic4 = ["original", "mean", "var", "std"]
     fig, plot = create_fig(2, 2)

     line(plot[0,1], arrArange(data_row), array_cal_each_id(y_arr, cal = "mean"), "mean", basic4[1], "blue", thickness)
     line(plot[1,0], arrArange(data_row), array_cal_each_id(y_arr, cal = "var"), "var", basic4[2], "blue", thickness)
     line(plot[1,1], arrArange(data_row), array_cal_each_id(y_arr, cal = "std"), "std", basic4[3], "blue", thickness)
     for i in range(data_row):
          if( str(itemId[i]) in damageList):
               line(plot[0,0], x_arr, y_arr[i], itemId[i], basic4[0], "red", 2)
               scatter(plot[0,1], i, array_cal_each_id(y_arr, cal = "mean")[i], "mean", basic4[1], "red", 5)
               scatter(plot[1,0], i, array_cal_each_id(y_arr, cal = "var")[i], "var", basic4[2], "red", 5)
               scatter(plot[1,1], i, array_cal_each_id(y_arr, cal = "std")[i], "std", basic4[3], "red", 5)
          else:
               line(plot[0,0], x_arr, y_arr[i], itemId[i], basic4[0], "blue", 0.1)
          print(" - drawing plot ... {}".format(i+1) + " / " + "{}".format(data_row) , end = "\r")
     print("\n")
     

     fig_name += "_basic4info"
     setting_fig(fig, fig_name)
     plot_show(fig, fig_name, save)

def plot_show(fig, name, save = True):
     manager = plt.get_current_fig_manager()
     #print("backend : " + str(matplotlib.get_backend()))
     if (matplotlib.get_backend() == 'TkAgg'):
          manager.window.state('zoomed')
     elif(matplotlib.get_backend() == 'wxAgg'):
          manager.frame.Maximize(True)
     elif(matplotlib.get_backend() == 'QT4Agg'):
          manager.window.showMaximized()

     if(save == True):
          plot_save(fig, name)
     
     plt.show()
     
def plot_save(fig, logName):
     fig.set_size_inches(19.2,10.8)
     project_root = os.path.abspath(os.path.dirname(__file__))
     dir_name = "plot_log"
     save_root = os.path.join(project_root, dir_name)
     if not os.path.exists(save_root):
          os.mkdir(save_root)

     nowDate = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(":","_")
     saveName = logName + nowDate
     save_full_root = os.path.join(save_root, saveName)
     fig.savefig(save_full_root, dpi=100, bbox_inches='tight')


