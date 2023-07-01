# 1. Imports:
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
# To run plt in jupyter or gg colab envirionment
# %matplotlib inline

from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10
from keras.models import Sequential
from keras.layers import LSTM,Dense,SimpleRNN
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import external_stock_data
import threading
import time
from keras.models import save_model
import joblib

def buildModel(stock, column, days, algorithm):
    # alert start
    print(f'runing {stock}_{column}_{algorithm}_model.h5')

    # get data
    df = external_stock_data.getStockDataToNow(stock, days)

    # 3. Analyze the closing prices from dataframe:
    df["Date"] = df.index

    # 4. Sort the dataset on date time and filter “Date” and “Close” columns:
    data=df.sort_index(ascending=True,axis=0)
    new_dataset=pd.DataFrame(index=range(0,len(df)),columns=['Date', column])

    for i in range(0,len(data)):
        new_dataset["Date"][i]=data['Date'][i]
        new_dataset[column][i]=data[column][i]

    # 5. Normalize the new filtered dataset:
    # get close price column
    new_dataset.index=new_dataset.Date
    new_dataset.drop("Date",axis=1,inplace=True)
    final_dataset=new_dataset.values

    # get range to train data and valid data
    train_data=final_dataset

    # scale close price to range 0,1
    scaler=MinMaxScaler(feature_range=(0,1))
    scaled_data=scaler.fit_transform(final_dataset)

    x_train_data,y_train_data=[],[]

    for i in range(60,len(train_data)):
        x_train_data.append(scaled_data[i-60:i,0])
        y_train_data.append(scaled_data[i,0])
        
    x_train_data,y_train_data=np.array(x_train_data),np.array(y_train_data)

    # 6. Build model using algorithm:
    if(algorithm == 'rnn'):
        x_train_data=np.reshape(x_train_data,(x_train_data.shape[0],x_train_data.shape[1],1))
        train_model = Sequential()
        train_model.add(SimpleRNN(units=50, return_sequences=True, input_shape=(x_train_data.shape[1], 1)))
        train_model.add(SimpleRNN(units=50))
        train_model.add(Dense(1))
        train_model.compile(loss='mean_squared_error', optimizer='adam')
        train_model.fit(x_train_data, y_train_data, epochs=1, batch_size=1, verbose=2)
        train_model.save(f"model/{stock}_{column}_{algorithm}_model.h5")
    elif(algorithm == 'xgboost'):
        train_model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
        train_model.fit(x_train_data, y_train_data)
        train_model.save_model(f"model/{stock}_{column}_{algorithm}_model.h5")
    else:
        x_train_data=np.reshape(x_train_data,(x_train_data.shape[0],x_train_data.shape[1],1))
        train_model = Sequential()
        train_model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_data.shape[1], 1)))
        train_model.add(LSTM(units=50))
        train_model.add(Dense(1))
        train_model.compile(loss='mean_squared_error', optimizer='adam')
        train_model.fit(x_train_data, y_train_data, epochs=1, batch_size=1, verbose=2)
        train_model.save(f"model/{stock}_{column}_{algorithm}_model.h5")

    #alert done
    print(f'done {stock}_{column}_{algorithm}_model.h5')
    return

def buildModelByLSTM(stock, column, days):
    result = buildModel(stock, column, days, 'lstm')
    return result

def buildModelByRNN(stock, column, days):
    result = buildModel(stock, column, days, 'rnn')
    return result

def labAsync(algorithm, coin, column):
    print(f"{algorithm}_{coin}_{column}")
    time.sleep(5)

# build some model
def buildAllModelsForSystem():
    coins = ['BTC-USD', 'ETH-USD', 'ADA-USD']
    algorithms = ['lstm', 'rnn', 'xgboost']
    #algorithms = ['xgboost']
    columns = ['Open', 'Close', 'Low', 'High']

    for coin in coins:
        threadsOfCoin = []

        # start thread of one coin
        for algorithm in algorithms:
            for column in columns:
                thread = threading.Thread(target=buildModel, args=(coin, column, 5*365, algorithm))
                threadsOfCoin.append(thread)
                thread.start()

        # await all above thread done
        for thread in threadsOfCoin:
            thread.join()


buildAllModelsForSystem()
