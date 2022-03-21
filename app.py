import os

from flask import Flask, jsonify, request



import SeperateData
import CleaningData

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home_page():
    return jsonify('testing api')

@app.route("/data", methods=['GET', 'POST'])
def data():
     directory_path = request.json['path']
     raw_data = pd.read_csv(directory_path +"\\"+"Dataset.csv")
     return jsonify(raw_data)
        

@app.route('/Preprocessing' , methods = ['POST'])
def PreprocessingData():
       directory_path = request.json['path']
       raw_data_name = request.json['file_name_rawdata']
       train_data_name = request.json['File_name_train']
       test_data_name = request.json['File_name_test']

       exportingData = SeperateData.DataProcessing()
       exportingData.ExportingData(directory_path, raw_data_name)

       raw_data = exportingData.CallingData(directory_path, raw_data_name)
       train, test = exportingData.SplittingData(raw_data)

       train.to_csv(directory_path + "\\" + train_data_name + ".csv", index=False, header=True)
       test.to_csv(directory_path + "\\" + test_data_name + ".csv", index=False, header=True)

       return jsonify("Final train and located at : {}".format(directory_path))



@app.route("/Training",methods = ['GET'])
def Training():
       directory_path = request.json["path"]
       train_data_name = request.json["File_name_train"]
       model_name = request.json["Model_name"]
       scaler_model = request.json["scaler_name"]

       train = SeperateData.DataProcessing().CallingData(directory_path, train_data_name)
       edaclass = CleaningData.Cleandata()
       train = edaclass.NullRemoval(train)
       train = edaclass.drop_duplicate(train)

       x_train, y_train = edaclass.scalling_train_data(train, directory_path, scaler_model)

       output = edaclass.modelLinear(x_train, y_train, directory_path, model_name)
       return jsonify("Value of R2 is equal to : {}".format(output))


@app.route("/Testing", methods=['GET'])
def Testing():
       directory_path = request.json["path"]
       test_data_name = request.json["File_name_test"]
       model_name = request.json["Model_name"]
       scaler_model = request.json["scaler_name"]
       test = SeperateData.DataProcessing().CallingData(directory_path, test_data_name)
       edaclass = CleaningData.Cleandata()

       test = edaclass.NullRemoval(test)
       test = edaclass.drop_duplicate(test)

       x_test, y_test = edaclass.scalling_test_data(test, directory_path, scaler_model)

       output_result = edaclass.prediction(x_test, y_test, directory_path, model_name)

       return jsonify("AARD Value is obatined {}".format(output_result["AARD"].loc["mean"]))


if __name__ == '__main__':
    app.run()










