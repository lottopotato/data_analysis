import matplotlib.pyplot as plt
import matplotlib

from numpy_process import *

def create_fig(dimension_x, dimension_y):
     if (dimension_x == None or dimension_y == None):
          flg, plot = plt.subplots()
     else:
          flg, plot = plt.subplots(dimension_x, dimension_y)
     return flg, plot

def setting_fig(fig, title):
     fig.canvas.set_window_title(title)
     fig.legend()

def scatter(plot, x_data, y_data, name, setColor, scale, option = "add"):
     if (option == "basic"):
          plt.scatter(x_data, y_data, color = setColor, s = scale)
          plt.title(name)
     else:
          plot.scatter(x_data, y_data, color = setColor, s= scale, label = name)
          plot.title(name)

def line(plot, x_data, y_data, name, setColor, option = "add"):
     if (option == "basic"):
          plt.plot(x_data, y_data, color = setColor)
          plt.title(name)
     else:
          plot.plot(x_data, y_data, color = setColor, label = name)
          plot.title(name)

def basic4info_fig(x_arr, y_arr):
     newX_arr = array_division_each_id(x_arr)

     mean_y = array_division_each_id(y_arr, "mean")
     var_y = array_division_each_id(y_arr, "var")
     std_y = array_division_each_id(y_arr, "std")

     fig, plot = create_fig(2,2)
     scatter(plot[0,0], x_arr, y_arr, "basic", "mediumvioletred", 0.1)
     scatter(plot[0,1], newX_arr, mean_y, "mean", "blue", 0.1)
     scatter(plot[1,0], newX_arr, var_y, "var", "red", 0.1)
     scatter(plot[1,1], newX_arr, std_y, "std", "green", 0.1)

     setting_fig(fig, "basic4info")
     plot_show()

def plot_show():
     manager = plt.get_current_fig_manager()
     if (matplotlib.get_backend() == 'TkAgg'):
          manager.window.state('zoomed')
     elif(matplotlib.get_backend() == 'wxAgg'):
          manager.frame.Maximize(True)
     elif(matplotlib.get_backend() == 'QT4Agg'):
          manager.window.showMaximized()
     
     plt.show()


