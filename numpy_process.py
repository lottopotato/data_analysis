import numpy as np

def expandDims(arr, axis):
     new_arr = np.expand_dims(arr, axis=axis)
     return new_arr

def arrArange(i):
     x_arr = np.arange(i)
     return x_arr

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

def array_cal_each_id(arr, cal = "none"):
     # calculator = ["None", "Mean", "Var", "Std"]
     newArr = np.zeros( int(len(arr[:])) )
     index = 0
     for i in range(len(arr[:])):
          if(cal == "none"):
               newArr = None
          elif(cal == "mean"):
               newArr[i] = get_mean(arr[:][index])
          elif(cal == "var"):
               newArr[i] = get_var(arr[:][index])
          elif(cal == "std"):
               newArr[i] = get_std(arr[:][index])
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

def reshape_2dto1d(arr):
     new_arr = arr.reshape( arr.shape[0] * arr.shape[1])
     return new_arr
     

     
