import os
import json
class DataGenerator():
    def __init__(self):
        self.pathtoCoreData = "./data/activities"
        self.coreData = self.loadCoreData()
        
    def loadCoreData(self):
        json_data = {}
        for filename in os.listdir(self.pathtoCoreData):
            if filename.endswith(".json"):
                with open(os.path.join(self.pathtoCoreData, filename), "r") as file:
                    data = json.load(file)
                    if isinstance(data, dict):
                        json_data.update(data)
        return json_data
    def countQuantityofEachKeys(self):
        for key, sub_array in self.coreData.items():
            count = len(sub_array)
            print(f"Key: {key}, Number of elements: {count}")
                
datagen = DataGenerator()

