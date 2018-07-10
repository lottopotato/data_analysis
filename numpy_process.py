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

def array_division_367(arr, cal = "none"):
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

     
