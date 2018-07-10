import matplotlib.pyplot as plt
import numpy as np

from model_read import *

# 367 
def load():
     data_root = root()

     data = read(data_root.T01_ROOT)

     x_data = []
     y_data = []

     for i in range(len(data)):
          y_data.extend(data[i]["srcData"])
          for j in range(len(data[i]["srcData"])):
               x_data.append(data[i]["itemId"])
     return x_data, y_data

def scatter(x_data, y_data):
     plt.scatter(x_data, y_data, s=0.1)
     plt.show()

def list2numpy(y_data):
     data_arr = np.asarray(y_data)

x, y = load()
y_arr = list2numpy(y[:367])
