# Imports:
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
# To run plt in jupyter or gg colab envirionment
# %matplotlib inline

from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
from keras.models import load_model
import external_stock_data
from datetime import date, timedelta
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import joblib


# Caluculate ROC form data like [[][]]
def calculate_roc(data):
    series = pd.Series(data.flatten())  
    roc = (series - series.shift(1)) / series.shift(1) * 100
    return roc.values.reshape(-1, 1)


def predict(stock, column, start_date, end_date, algorithm):
    # get data
    new_start_date = date.strftime(pd.to_datetime(start_date) - timedelta(60), '%Y-%m-%d')
    df = external_stock_data.getStockData(stock, new_start_date, end_date)
    df["Date"] = df.index

    # Get predict Close alternative for ROC
    temp = ""
    if (column == "ROC"):
        temp = "ROC"
        column = 'Close'
    

    # sort data
    data=df.sort_index(ascending=True,axis=0)
    new_data=pd.DataFrame(index=range(0,len(df)),columns=['Date',column])

    # get essential column
    for i in range(0,len(data)):
        new_data["Date"][i]=data['Date'][i]
        new_data[column][i]=data[column][i]

    new_data.index=new_data.Date
    new_data.drop("Date",axis=1,inplace=True)

    # scale data
    scaler=MinMaxScaler(feature_range=(0,1))
    scaler.fit_transform(new_data.values)

    # get data to predict
    inputs=new_data.values
    inputs=inputs.reshape(-1,1)
    inputs=scaler.transform(inputs)
    

    X_test=[]
    for i in range(60, inputs.shape[0]):
        X_test.append(inputs[i-60:i,0])
    X_test=np.array(X_test)
    X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
    

    # load model to predict
    model=load_model(f"model/{stock}_{column}_{algorithm}_model.h5")

    # predict
    pred_price=model.predict(X_test)
    pred_price=scaler.inverse_transform(pred_price)

    # scale one day because 60 previous days will predict next day
    pred = new_data[61:]

    # Create a new row of data
    newDate = date.strftime(pd.to_datetime(pred.index[-1]) + timedelta(1), '%Y-%m-%d')
    new_row = pd.DataFrame(index=pd.to_datetime([newDate]))

    # Append the new row to the DataFrame
    pred = pd.concat([pred, new_row])
   

    # Calculate ROC values
    if temp == "ROC":
        roc_values = calculate_roc(pred_price)
       
        pred_price = roc_values
        print("pred[column]",pred_price)
        
    
    # return result
    pred["Predictions"] = pred_price
    return pred

def predictNextFrames(predResult, stock, column, algorithm, nFrames):
    # scale data
    scaler=MinMaxScaler(feature_range=(0,1))
    scaler.fit_transform(predResult.values)

    # get data to predict
    inputs=[predResult.values]
    inputs=inputs.reshape(-1,1)
    inputs=scaler.transform(inputs)

    X_test=[]
    X_test.append(inputs[inputs.shape[0]-60:inputs.shape[0],0])
    X_test=np.array(X_test)

    X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

    # load model to predict
    model=load_model(f"model/{stock}_{column}_{algorithm}_model.h5")

    # predict
    print(X_test)
    pred_price=model.predict(X_test)
    pred_price=scaler.inverse_transform(pred_price)

    # scale one day because 60 previous days will predict next day
    print(pred_price)
    return pred_price
        


def predictByLSTM(stock, column, start_date, end_date):
    result = predict(stock, column, start_date, end_date, 'lstm')
    return result

def predictByRNN(stock, column, start_date, end_date):
    result = predict(stock, column, start_date, end_date, 'rnn')
    return result

def predictByXGBoost(stock, column, start_date, end_date):
    result = predict(stock, column, start_date, end_date, 'rnn')
    return result
