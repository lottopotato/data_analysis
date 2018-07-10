import numpy as np

from data_type import dataType

dataType = dataType()

def list2numpy(data):
     data_arr = np.asarray(data)
     return data_arr

def get_mean(arr):
     arr_mean = arr.mean()
     return arr_mean

def get_var(arr):
     arr_var = arr.var()
     return arr_var

def get_std(arr):
     arr_std = arr.std()
     return arr_std

def array_division_each_id(arr, cal = "none"):
     # calculator = ["None", "Mean", "Var", "Std"]
     newArr = np.zeros( int(len(arr) / dataType.SRC_len) )
     index = 0
     for i in range(len(arr)):
          if i % dataType.SRC_len == 0:
               if(cal == "none"):
                    newArr[index] = arr[i]
               elif(cal == "mean"):
                    newArr[index] = get_mean(arr[i : i + dataType.SRC_len])
               elif(cal == "var"):
                    newArr[index] = get_var(arr[i : i + dataType.SRC_len])
               elif(cal == "std"):
                    newArr[index] = get_std(arr[i : i + dataType.SRC_len])
               else:
                    print(" error from numpy_process.py. array_division_367 ")
               index += 1
     return newArr

def division_pairArr(pairArr):
     arr_len = pairArr.shape[0]
     newX_arr = np.zeros(arr_len)
     newY_arr = np.zeros(arr_len)
     for i in range(arr_len):
          newX_arr[i] = pairArr[i,0]
          newY_arr[i] = pairArr[i,1]
     return newX_arr, newY_arr

def sampling_arr(x_arr, y_arr, index):
     newX_arr = x_arr[0+(index*dataType.SRC_len) : dataType.SRC_len + (index*dataType.SRC_len)]
     newY_arr = y_arr[0+(index*dataType.SRC_len) : dataType.SRC_len + (index*dataType.SRC_len)]
     return newX_arr, newY_arr

def merge_pairArr(x_arr, y_arr):
     xd = len(x_arr)
     newArr = np.zeros([xd, 2])
     for i in range(newArr.shape[0]):
          newArr[i, 0] = x_arr[i]
          newArr[i, 1] = y_arr[i]
     return newArr

def merge_arr(x_arr, y_arr):
     xd = len(x_arr)
     yd = len(y_arr)
     newArr = np.zeros([xd, yd])
     

     
