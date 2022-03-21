import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pickle
class Cleandata:
    def __init__(self):
        pass

    def NullRemoval(self,data):
        self.data = data
        self.data.dropna(inplace=True)
        return self.data

    def drop_duplicate(self,data):
        self.data = data
        self.data.drop_duplicates(inplace= True)
        return self.data

    def scalling_train_data(self,data,path,filename):
        self.data = data
        self.features = self.data.drop("price",axis=1)
        self.label = self.data[["price"]]
        self.path = path
        self.filename = filename + ".sav"
        scaler = StandardScaler()
        self.scalleddata = scaler.fit_transform(self.features)
        pickle.dump(scaler, open(self.path +"//"+self.filename, "wb"))
        return self.scalleddata,self.label

    def scalling_test_data(self,data,path,filename):
        self.data = data
        self.features = self.data.drop("price", axis=1)
        self.label = self.data["price"]
        scaler = pickle.load(open(path+"//" + filename + ".sav" , "rb"))
        self.scalleddata = scaler.transform(self.features)
        return self.scalleddata, self.label

    def modelLinear(self,features,label,path,filename):
        self.model = LinearRegression()
        self.model.fit(features,label)
        self.path = path
        self.filename = filename + ".sav"
        pickle.dump(self.model ,open(self.path+"//"+self.filename , "wb"))
        self.score = self.model.score(features,label)
        return self.score

    def prediction(self,features,label,path,filename):
        self.path = path
        self.filename = filename + ".sav"
        self.model = pickle.load(open(self.path+ "//"+self.filename , "rb"))
        self.output = self.model.predict(features)
        result = pd.DataFrame()
        result["Predicted Price"] = self.output.flatten()
        result["Actual price"] = label
        result["Residuals"] = result["Actual price"] - result["Predicted Price"]
        result["AARD"] = abs(result["Residuals"]) / result["Actual price"]
        result_description = result.describe()
        return result_description


