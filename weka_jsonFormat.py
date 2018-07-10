def weka_json_format():
     front_content = ("{\"item\":{",
                      "\n\"relation\":\"parts\",",
                      "\n\"attributes\":{",
                      "\n    \"itemId\":",
                      "\n    \"srcData\":",
                      "\n },",
                      "\n\"data\":[")
     front_str = ''.join(front_content)

     contents = "\n some content"

     last_str = "\n]}}"

     return front_str, last_str
