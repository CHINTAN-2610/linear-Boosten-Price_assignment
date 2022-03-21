import pandas as pd

from sklearn.datasets import load_boston

class Dataset:

    def __init__(self,path):
        self.path = path


    def LoadData(self,filename):
        '''

             Load dataset and store dataset at
             location defined in path and file name

             '''
        self.filename = filename
        data = load_boston()
        bos = pd.DataFrame(data["data"])
        bos.columns = data.feature_names
        bos["price"] = data["target"]
        bos.to_csv(self.path+"\\"+self.filename, index=False, header=True)
        #print("Dataset is stored at location : {}".format(self.path + self.filename))