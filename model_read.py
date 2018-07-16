import os, json
import matplotlib.pyplot as plt

from weka_Format import *

class root():
     def __init__(self):
          self.QGdata_ROOT = "C:/Users/khh/Documents/타키온/data/QG데이터"
          self.model_ID_1 = "32J_06_T06"
          self.model_ID_2 = "SHAFT_OP180_3"

          self.model_ROOT_1 = self.QGdata_ROOT + "/" + self.model_ID_1
          self.model_ROOT_2 = self.QGdata_ROOT + "/" + self.model_ID_2

          self.model_T06 = "inspectionRawData20170701_20170731.json"
          self.model_T01 = "inspectionRawData20170801_20170811.json"
          self.model_T07 = "inspectionRawData20170712_20170721.json"

          self.T01_ROOT = os.path.join(self.model_ROOT_2 + "/T01/", self.model_T01)
          self.T06_ROOT = os.path.join(self.model_ROOT_1, self.model_T06)
          self.T07_ROOT = os.path.join(self.model_ROOT_2 + "/T07/", self.model_T07)
          self.test = os.path.join(self.model_ROOT_2 + "/T07/", "test.json")

def read(model_root):
     data = []
     sample = {}
     dataList = []
     with open(model_root, "rb") as f:
          data = json.load(f)
          for i in range(len(data)):
               insert_dict(sample, data[i]["itemId"], "itemId")
               insert_dict(sample, data[i]["srcData"], "srcData")
               print( " data read - {} ".format(i+1) + " / " + " {} ".format(len(data)), end="\r")
               dataList.append(sample)
               sample = {}
          print("\n data read done ")
          print(" ========================== ")
     f.close()
     return dataList

def save_arff(dataList, fileName):
     arff_str = weka_data_format()
     
     dir_root = "./temp_arff"
     save_root = os.path.join(dir_root, fileName)
     if not (os.path.exists(dir_root)):
          os.mkdir(dir_root)

     temp = []
     with open(save_root, 'w') as f:
          f.write(arff_str)     
          for i in range(len(dataList)):
               f.write(str(dataList[i]["itemId"]))
               f.write(",")
               srcData = ','.join(list(map(str, dataList[i]["srcData"])))
               f.write(srcData)
               f.write("\n")

     f.close()

def insert_dict(new_dic, val, key):
     if not key in new_dic:
          new_dic[key] = val
     else:
          new_dic[key].append(val)

def load(dataName, saveOp = True):
     data_root = root()
     if (dataName == "T01"):
          data = read(data_root.T01_ROOT)
     elif(dataName == "T06" ):
          data = read(data_root.T06_ROOT)
     elif(dataName == "T07" ):
          data = read(data_root.T07_ROOT)
     elif(dataName == "test"):
          data = read(data_root.test)
     else:
          print( "error from dataName")
          return False

     if(saveOp == True):
          save_arff(data, "temp_arff.arff")

     x_data = []
     y_data = []

     for i in range(len(data)):
          x_data.append(data[i]["itemId"])
          y_data.append(data[i]["srcData"])       
     return x_data, y_data



