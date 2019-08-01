

import os
from flask import Flask, request, jsonify, render_template, redirect, url_for, send_file
import pandas as pd
import numpy
import pickle
import json
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.style.use('ggplot')




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

        elif modelSelected.find('SKLearn_MultipleRegression') > 0:

            # get new X
            datadf['lottoX'] = datadf['First Number'].astype(str) +  datadf['Second Number'].astype(str) +  datadf['Third Number'].astype(str) 
            dataLotto = datadf[['lottoX']].astype(float)
            newX = numpy.asarray(dataLotto['lottoX'])
            newX = newX[-200:]
            newX = newX.reshape(1, -1)
            newX


            #load model
            import pickle
            pkl_filename = "MLModels/SKLearn_MultipleRegression_AP_R100_R000.pkl"
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

        elif modelSelected.find('SKLearn_LinearRegression') > 0:
           
            datadf['lottoNum'] = datadf['First Number'].astype(str) +  datadf['Second Number'].astype(str) +  datadf['Third Number'].astype(str) 
            datadf['lottoX'] = datadf['lottoNum'].shift(-1)
            datadf = datadf.drop('Draw Date', axis=1)
            datadf = datadf.drop('First Number', axis=1)
            datadf = datadf.drop('Second Number', axis=1)
            datadf = datadf.drop('Third Number', axis=1)
            datadf = datadf.dropna()
            datadf['lottoNum'] = datadf['lottoNum'].astype(int)
            datadf['lottoX'] = datadf['lottoX'].astype(int)
            df2 = pd.get_dummies(datadf)
            X = df2[['lottoX','Draw Schedule_Evening', 'Draw Schedule_Morning']]
            y = df2['lottoNum'].values.reshape(-1, 1)

            X = X[-1:]
            
            


            ## Load saved model
            import pickle
            pkl_filename = "MLModels/SKLearn_LinearRegression_MornEve.pkl"
            with open(pkl_filename, 'rb') as file:
                pickle_model = pickle.load(file)

            #predict
            Ypredict = pickle_model.predict(X)
            Ypredict = int(numpy.round(Ypredict,0))
            Ypredict = str(Ypredict)

            firstY = int(Ypredict[0])
            secondY = int(Ypredict[1])
            thirdY = int(Ypredict[2])

            list_of_files = getListOfTrainedModels()

            return render_template ('home.html', firstY=firstY, secondY=secondY, thirdY=thirdY, list_of_files = list_of_files)


        elif modelSelected.find('SKLearn_Ridge') > 0:
           
            datadf['lottoNum'] = datadf['First Number'].astype(str) +  datadf['Second Number'].astype(str) +  datadf['Third Number'].astype(str) 
            datadf['lottoX'] = datadf['lottoNum'].shift(-1)
            datadf = datadf.drop('Draw Date', axis=1)
            datadf = datadf.drop('First Number', axis=1)
            datadf = datadf.drop('Second Number', axis=1)
            datadf = datadf.drop('Third Number', axis=1)
            datadf = datadf.dropna()
            datadf['lottoNum'] = datadf['lottoNum'].astype(int)
            datadf['lottoX'] = datadf['lottoX'].astype(int)
            df2 = pd.get_dummies(datadf)
            X = df2[['lottoX','Draw Schedule_Evening', 'Draw Schedule_Morning']]
            y = df2['lottoNum'].values.reshape(-1, 1)

            X = X[-1:]
            
            


            ## Load saved model
            import pickle
            pkl_filename = "MLModels/SKLearn_Ridge_MornEve.pkl"
            with open(pkl_filename, 'rb') as file:
                pickle_model = pickle.load(file)

            #predict
            Ypredict = pickle_model.predict(X)
            Ypredict = int(numpy.round(Ypredict,0))
            Ypredict = str(Ypredict)

            firstY = int(Ypredict[0])
            secondY = int(Ypredict[1])
            thirdY = int(Ypredict[2])

            list_of_files = getListOfTrainedModels()

            return render_template ('home.html', firstY=firstY, secondY=secondY, thirdY=thirdY, list_of_files = list_of_files)

        elif modelSelected.find('SKLearn_Lasso') > 0:
           
            datadf['lottoNum'] = datadf['First Number'].astype(str) +  datadf['Second Number'].astype(str) +  datadf['Third Number'].astype(str) 
            datadf['lottoX'] = datadf['lottoNum'].shift(-1)
            datadf = datadf.drop('Draw Date', axis=1)
            datadf = datadf.drop('First Number', axis=1)
            datadf = datadf.drop('Second Number', axis=1)
            datadf = datadf.drop('Third Number', axis=1)
            datadf = datadf.dropna()
            datadf['lottoNum'] = datadf['lottoNum'].astype(int)
            datadf['lottoX'] = datadf['lottoX'].astype(int)
            df2 = pd.get_dummies(datadf)
            X = df2[['lottoX','Draw Schedule_Evening', 'Draw Schedule_Morning']]
            y = df2['lottoNum'].values.reshape(-1, 1)

            X = X[-1:]
            
            


            ## Load saved model
            import pickle
            pkl_filename = "MLModels/SKLearn_Lasso_MornEve.pkl"
            with open(pkl_filename, 'rb') as file:
                pickle_model = pickle.load(file)

            #predict
            Ypredict = pickle_model.predict(X)
            Ypredict = int(numpy.round(Ypredict,0))
            Ypredict = str(Ypredict)

            firstY = int(Ypredict[0])
            secondY = int(Ypredict[1])
            thirdY = int(Ypredict[2])

            list_of_files = getListOfTrainedModels()

            return render_template ('home.html', firstY=firstY, secondY=secondY, thirdY=thirdY, list_of_files = list_of_files)  

        elif modelSelected.find('SKLearn_ElasticNet') > 0:
           
            datadf['lottoNum'] = datadf['First Number'].astype(str) +  datadf['Second Number'].astype(str) +  datadf['Third Number'].astype(str) 
            datadf['lottoX'] = datadf['lottoNum'].shift(-1)
            datadf = datadf.drop('Draw Date', axis=1)
            datadf = datadf.drop('First Number', axis=1)
            datadf = datadf.drop('Second Number', axis=1)
            datadf = datadf.drop('Third Number', axis=1)
            datadf = datadf.dropna()
            datadf['lottoNum'] = datadf['lottoNum'].astype(int)
            datadf['lottoX'] = datadf['lottoX'].astype(int)
            df2 = pd.get_dummies(datadf)
            X = df2[['lottoX','Draw Schedule_Evening', 'Draw Schedule_Morning']]
            y = df2['lottoNum'].values.reshape(-1, 1)

            X = X[-1:]
            
            


            ## Load saved model
            import pickle
            pkl_filename = "MLModels/SKLearn_ElasticNet_MornEve.pkl"
            with open(pkl_filename, 'rb') as file:
                pickle_model = pickle.load(file)

            #predict
            Ypredict = pickle_model.predict(X)
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
        return redirect (url_for("getDataFromCSV"))

    return render_template ('data.html')




@app.route('/data')
def getDataFromCSV():
    
    # Create a reference the CSV file desired
    csv_path = "dailyData/dailyData.csv"

    # Read the CSV into a Pandas DataFrame
    df = pd.read_csv(csv_path)
    

    return render_template ('data.html', tables=[df.to_html(classes = 'data')], titles = df.columns.values)


@app.route("/graph", methods=['GET', 'POST'])
def graph():
    if request.method == 'POST':
        # Which graph is selected?
        graphlist = request.form.get('graphlist')
        graphlist = str(graphlist)
        print(graphlist)

        if graphlist== 'FirstPosition':

            # Create a reference the CSV file desired
            csv_path = "dailyData/dailyData.csv"
            #Read the CSV into a Pandas DataFrame
            df = pd.read_csv(csv_path)
            

            #mpl.rc("figure", figsize=(3, 3))
            plt.figure(figsize=(5,5))
            ax = sns.countplot(y="First Number", data=df)
            plt.title("Numbers First Positions")
            plt.xlabel("Occurence")
            plt.ylabel("Lotto Number")

            img = BytesIO()
            plt.savefig(img)
            img.seek(0)

            return send_file(img, mimetype = 'image/png') 

        elif graphlist== 'SecondPosition':

            # Create a reference the CSV file desired
            csv_path = "dailyData/dailyData.csv"
            #Read the CSV into a Pandas DataFrame
            df = pd.read_csv(csv_path)
            

            #mpl.rc("figure", figsize=(3, 3))
            plt.figure(figsize=(5,5))
            ax = sns.countplot(y="Second Number", data=df)
            plt.title("Numbers Second Positions")
            plt.xlabel("Occurence")
            plt.ylabel("Lotto Number")

            img = BytesIO()
            plt.savefig(img)
            img.seek(0)

            return send_file(img, mimetype = 'image/png') 

        elif graphlist== 'ThirdPosition':

            # Create a reference the CSV file desired
            csv_path = "dailyData/dailyData.csv"
            #Read the CSV into a Pandas DataFrame
            df = pd.read_csv(csv_path)
            

            #mpl.rc("figure", figsize=(3, 3))
            plt.figure(figsize=(5,5))
            ax = sns.countplot(y="Third Number", data=df)
            plt.title("Numbers Third Positions")
            plt.xlabel("Occurence")
            plt.ylabel("Lotto Number")

            img = BytesIO()
            plt.savefig(img)
            img.seek(0)

            return send_file(img, mimetype = 'image/png') 


        elif graphlist== 'Occurances':

            # Create a reference the CSV file desired
            csv_path = "dailyData/dailyData.csv"
            #Read the CSV into a Pandas DataFrame
            df = pd.read_csv(csv_path)
            df = df[['First Number', 'Second Number', 'Third Number']]
            new_df_col_all= df[['First Number', 'Second Number' ,'Third Number']]
            sns.countplot(x="variable", hue="value", data=pd.melt(new_df_col_all))
            plt.title("Occurence in all three positions")
            img = BytesIO()
            plt.savefig(img)
            img.seek(0)

            return send_file(img, mimetype = 'image/png') 

        elif graphlist == 'HotNumbers':
            # Create a reference the CSV file desired
            csv_path = "dailyData/dailyData.csv"
            # Read the CSV into a Pandas DataFrame
            df = pd.read_csv(csv_path)
            df = df[['First Number', 'Second Number', 'Third Number']]
            new_df_col_all= df[['First Number', 'Second Number' ,'Third Number']]

            #Count of occurances  in first position
            First= new_df_col_all.groupby(['First Number']).count().reset_index()                
            First = First.drop(columns=['First Number', 'Third Number'])
            First = First.rename(columns={'Second Number': 'First'}).reset_index()
            #Count of occurances  in second position
            Second= new_df_col_all.groupby(['Second Number']).count().reset_index()
            Second = Second.drop(columns=['Second Number', 'Third Number'])
            Second = Second.rename(columns={'First Number': 'Second'}).reset_index()
            #Count of occurances  in third position
            Third= new_df_col_all.groupby(['Third Number']).count().reset_index()                
            Third = Third.drop(columns=['Third Number', 'Second Number'])
            Third = Third.rename(columns={'First Number': 'Third'}).reset_index()
            #Mergec Columns
            hotN =  pd.merge(First, Second,  on='index', how='inner')                
            hotN =  pd.merge(hotN, Third,  on='index', how='inner')
            #Get total
            hotN['Total'] = hotN['First'].sum()
            #Get percentages
            hotN['PFirst'] = hotN['First']/hotN['Total']
            hotN['PSecond'] = hotN['Second']/hotN['Total']
            hotN['PThird'] = hotN['Third']/hotN['Total']
            hotN = hotN.rename(columns = {'index': 'LottoNum'}).reset_index()
            #Get index of max values
            hotNarray = hotN.idxmax().values
            #declare an array
            valNarray = []
            #get hot number and % of occurance for 1st position
            photF= hotN[(hotN['LottoNum'] == hotNarray[6])]
            photF = photF[['LottoNum', 'PFirst']]
            valNarray.append(photF.values[0])
            #get hot number and % of occurance for 2nd position
            photS= hotN[(hotN['LottoNum'] == hotNarray[7])]
            photS = photS[['LottoNum', 'PSecond']]
            valNarray.append(photS.values[0])
            #get hot number and % of occurance for 3rd position
            photT= hotN[(hotN['LottoNum'] == hotNarray[8])]
            photT = photT[['LottoNum', 'PThird']]
            valNarray.append(photT.values[0])
            #turn to dataframe
            dfhotNum = pd.DataFrame(valNarray)


            # plot the hot number graph
            plt.figure(figsize=(5,5))
            dfhotNum.plot(kind='bar', x =0, y = 1,alpha=0.75, rot=0, legend=None)
            plt.title("Percentage of Most Occuring Daily 3 Numbers")
            #plt.xlabel("Daily 3 Number by Position")
            plt.ylabel("Percent")
            plt.xlabel("")
            img = BytesIO()
            plt.savefig(img)
            img.seek(0)

            return send_file(img, mimetype = 'image/png') 
        else:
            return render_template ('graph.html')

    return render_template ('graph.html')

if __name__ == "__main__":
    app.run(debug=True)
