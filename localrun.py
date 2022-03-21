import os

from flask import Flask, jsonify, request



import SeperateData
import CleaningData


directory_path = os.getcwd()
raw_data_name = "Dataset"
train_data_name = "Train"
test_data_name = "Test"
model_name = "linear"
scaler_model = "scaler"

exportingData = SeperateData.DataProcessing()
exportingData.ExportingData(directory_path, raw_data_name)

raw_data = exportingData.CallingData(directory_path, raw_data_name)
train, test = exportingData.SplittingData(raw_data)

train.to_csv(directory_path + "\\" + train_data_name + ".csv", index=False, header=True)
test.to_csv(directory_path + "\\" + test_data_name + ".csv", index=False, header=True)

# return jsonify("Final train and located at : {}".format(directory_path))
print("Final train and located at : {}".format(directory_path))

train = SeperateData.DataProcessing().CallingData(directory_path, train_data_name)
edaclass = CleaningData.Cleandata()
train = edaclass.NullRemoval(train)
train = edaclass.drop_duplicate(train)

x_train, y_train = edaclass.scalling_train_data(train,directory_path,scaler_model)

output = edaclass.modelLinear(x_train, y_train, directory_path, model_name)
print("Value of R2 is equal to : {}".format(output))

test = SeperateData.DataProcessing().CallingData(directory_path, test_data_name)
edaclass = CleaningData.Cleandata()

test = edaclass.NullRemoval(test)
test = edaclass.drop_duplicate(test)

x_test, y_test = edaclass.scalling_test_data(test,directory_path,scaler_model)

output_result = edaclass.prediction(x_test, y_test, directory_path, model_name)

print(output_result)