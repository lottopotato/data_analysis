def weka_data_format():
     # for json
     """
     front_content = ("{\"header\":{",
                      "\n\"relation\":\"parts\",",
                      "\n\"attributes\":{",
                      "\n    \"itemId\":",
                      "\n    \"srcData\":",
                      "\n },",
                      "\n\"data\":[")
                      """
     # for arff
     front_str = "@relation item\n@attribute parts NUMERIC\n"

     data_attribute = []
     for i in range(367):
          data_attribute.append("@attribute srcData{} NUMERIC \n".format(i+1))
     data_str = ''.join(data_attribute)

     contents = "\n some content"

     last_str = "\n@data\n"

     arff_str = front_str + data_str + last_str
     return arff_str


