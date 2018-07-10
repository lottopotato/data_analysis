import os, json
import matplotlib.pyplot as plt

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

def read(model_root):
     data = []
     sample = {}
     dataList = []
     with open(model_root, "rb") as f:
          data = json.load(f)
          for i in range(len(data)):
               insert_dict(sample, data[i]["itemId"], "itemId")
               insert_dict(sample, data[i]["srcData"], "srcData")

               dataList.append(sample)
               sample = {}

     return dataList

def insert_dict(new_dic, val, key):
     if not key in new_dic:
          new_dic[key] = val
     else:
          new_dic[key].append(val)

