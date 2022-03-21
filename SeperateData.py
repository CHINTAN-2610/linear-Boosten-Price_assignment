import os
import pandas as pd
from sklearn.model_selection import train_test_split

import ExtractDataset





class DataProcessing:

    def __init__(self):
        pass

    def ExportingData(self,directory_path,file_name):
        self.file_name = file_name + ".csv"
        self.directory_path = directory_path
        data = ExtractDataset.Dataset(directory_path)
        data.LoadData(self.file_name)

    def CallingData(self,directory_path,file_name):
        self.directory_path = directory_path
        self.file_name = file_name + ".csv"
        for file in os.listdir(self.directory_path):
            if file == self.file_name:
                raw_data = pd.read_csv(file)
                break
        return raw_data

    def SplittingData(self,raw_data):
        self.raw_data = raw_data
        train, test = train_test_split(raw_data,test_size = 0.30 , random_state=42)
        return train, test




