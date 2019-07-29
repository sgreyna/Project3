

import os
from flask import Flask, request, jsonify, render_template, redirect, url_for
import pandas as pd
import numpy
import pickle
import json



from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from tensorflow.keras.models import load_model



app = Flask(__name__)


def getListOfTrainedModels():
    path = os.getcwd()+"/MLModels"
    list_of_files = {}

    for filename in os.listdir(path):
        list_of_files[filename] = filename[0]

    return list_of_files



def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


@app.route("/",  methods =['GET', 'POST'] )
def main():
    firstY = 0
    secondY = 0
    thirdY = 0

    list_of_files = getListOfTrainedModels()
 

    return render_template ('home.html', firstY=firstY, secondY=secondY, thirdY=thirdY, list_of_files = list_of_files)

@app.route("/clear",  methods =['GET', 'POST'] )
def clear():

    return redirect (url_for("main"))

@app.route("/predictTrainedModelRoute", methods=['POST'])
def predictTrainedModel():
    if request.method == 'POST':
        print('complete')

        #import dailyData
        datadf = pd.read_csv("dailyData/dailyData.csv") 

        # Which model is selected?
        modelSelection = request.form.get('trainedModelList')
        modelSelected = str(modelSelection)
        print(modelSelected)
   
        if modelSelected.find('Keras_Sequential_30') > 0:


            ################################################
            #### First Number
            ################################################

            df = datadf['First Number']

            lottoNum = numpy.asarray(df)
            num = lottoNum[-30:]

            from tensorflow.keras.models import load_model
            model = load_model("MLModels/Keras_Sequential_30_P1_T100_A000.h5")

            x_input = num
            x_input = x_input.reshape((1, 30, 1))
            firstY = model.predict(x_input, verbose=0)
            firstY = int(round(firstY[0][0]))
            print(firstY)


            ################################################
            #### Second Number
            ################################################

            df = datadf['Second Number']

            lottoNum = numpy.asarray(df)
            num = lottoNum[-30:]

      
            model = load_model("MLModels/Keras_Sequential_30_P2_T100_A000.h5")

            x_input = num
            x_input = x_input.reshape((1, 30, 1))
            secondY = model.predict(x_input, verbose=0)
            secondY = int(round(secondY[0][0]))
            print(secondY)

            ################################################
            #### Third Number
            ################################################

            df = datadf['Third Number']

            lottoNum = numpy.asarray(df)
            num = lottoNum[-30:]

      
            model = load_model("MLModels/Keras_Sequential_30_P3_T100_A000.h5")

            x_input = num
            x_input = x_input.reshape((1, 30, 1))
            thirdY = model.predict(x_input, verbose=0)
            thirdY = int(round(thirdY[0][0]))
            print(thirdY)

            #model_loss, model_accuracy =  model.evaluate(X,y,verbose=1)
            #print({model_accuracy})

            list_of_files = getListOfTrainedModels()

            return render_template ('home.html', firstY=firstY, secondY=secondY, thirdY=thirdY, list_of_files = list_of_files, modelSelected= modelSelected)

        elif modelSelected.find('pSKLearn_MultipleRegression') > 0:

            # get new X
            datadf['lottoX'] = datadf['First Number'].astype(str) +  datadf['Second Number'].astype(str) +  datadf['Third Number'].astype(str) 
            dataLotto = datadf[['lottoX']].astype(float)
            newX = numpy.asarray(dataLotto['lottoX'])
            newX = newX[-200:]
            newX = newX.reshape(1, -1)
            newX


            #load model
            import pickle
            pkl_filename = "MLModels/pSKLearn_MultipleRegression_AP_R100_R000.pkl"
            with open(pkl_filename, 'rb') as file:
                pickle_model = pickle.load(file)

            #predict
            Ypredict = pickle_model.predict(newX)
            Ypredict = int(numpy.round(Ypredict,0))
            Ypredict = str(Ypredict)

            firstY = int(Ypredict[0])
            secondY = int(Ypredict[1])
            thirdY = int(Ypredict[2])

            list_of_files = getListOfTrainedModels()

            return render_template ('home.html', firstY=firstY, secondY=secondY, thirdY=thirdY, list_of_files = list_of_files)



        else:
            firstY = 0
            secondY = 0
            thirdY = 0

            list_of_files = getListOfTrainedModels()

            return render_template ('home.html', firstY=firstY, secondY=secondY, thirdY=thirdY, list_of_files = list_of_files)

        


@app.route("/predictAndTrainRoute", methods=['POST'])
def predictAndTrain():
     # Add code here
     return render_template ('home.html', firstY=firstY, secondY=secondY, thirdY=thirdY, model=model)




@app.route("/addToCSVRoute", methods=['POST'])
def addToCsv():
    if request.method =='POST':
        csvData = [];
        csv_path = "dailyData/dailyData.csv"

        # newData = pd.DataFrame({'Draw Date': csvData[:,3], 'Draw Sequence': csvData[:,4],'First Number': csvData[:,3],'Second Number': csvData[:,3],'Third Number': csvData[:,3]})
        newData = pd.DataFrame([{'Draw Date':request.form['drawDate'], 'Draw Sequence': request.form['drawSequence'],'First Number': request.form['firstNumberActual'],'Second Number': request.form['secondNumberActual'],'Third Number': request.form['thirdNumberActual']}])
        print(newData)

        newData.to_csv(csv_path, mode='a', header = False, index = False)

        # alert('Data Save')
        # return render_template('home.html')


    return render_template ('data.html')




@app.route('/data')
def getDataFromCSV():
    
    # Create a reference the CSV file desired
    csv_path = "dailyData/dailyData.csv"

    # Read the CSV into a Pandas DataFrame
    df = pd.read_csv(csv_path)

    return render_template ('data.html', tables=[df.to_html(classes = 'data')], titles = df.columns.values)


@app.route('/graph')
def graph():

    # Create a reference the CSV file desired
    csv_path = "dailyData/dailyData.csv"

    # Read the CSV into a Pandas DataFrame
    df = pd.read_csv(csv_path)
    df = df[['First Number', 'Second Number', 'Third Number']]
    df = df.to_dict() 

    print(df)

    return render_template ('graph.html', df=df)


if __name__ == "__main__":
    app.run(debug=True)
