import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math

from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.models import Sequential, load_model

import os


def create_dataset(dataset,look_back=2):
    Train_X, Train_Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back)]
        Train_X.append(a)
        Train_Y.append(dataset[i + look_back])
    Train_X = np.array(Train_X)
    Train_Y = np.array(Train_Y)
    return  Train_X,Train_Y



def Normalize(list):
    list = np.array(list)
    low, high = np.percentile(list, [0, 100])
    delta = high - low
    if delta != 0:
        for i in range(0, len(list)):
            list[i] = (list[i]-low)/delta
    return  list,low,high

def FNoramlize(list,low,high):
    delta = high - low
    if delta != 0:
        for i in range(0, len(list)):
            list[i] = list[i]*delta + low
    return list
dataset = np.arange(10,dtype='float64')
dataset,train_low,train_high = Normalize(dataset)
print(dataset)

Trian_X,Train_Y = create_dataset(dataset)
print(Trian_X)
print(Train_Y)
Trian_X = np.reshape(Trian_X, (Trian_X.shape[0], Trian_X.shape[1], 1))
print(Trian_X.shape)
Trian_Y= np.reshape(Train_Y,(Train_Y.shape[0],1))

#进行LSTM训练
# model = Sequential()
# model.add(LSTM(4, input_shape=(Trian_X.shape[1],Trian_X.shape[2])))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(Trian_X, Train_Y, epochs=1000, batch_size=1, verbose=2)
# model.save(os.path.join("DATA","LSTMTeach" + ".h5"))

#
model = load_model(os.path.join("DATA","LSTMTeach" + ".h5"))
# Test_X = [1,2,700,800]
# Test_X = np.array(Test_X,dtype='float64')
# Test_X = Normalize(Test_X)[0]
Test_X = [[0.,0.00125156],[0.87484355,1.]]
Test_X = np.array(Test_X)
Test_X = np.reshape(Test_X,(Test_X.shape[0],Test_X.shape[1],1))
# print(Test_X)
Y_hat = model.predict(Test_X)
Y_hat = np.array(Y_hat)
Y_hat = np.reshape(Y_hat,Y_hat.shape[0])
print(Y_hat.shape)
Y_hat = FNoramlize(Y_hat,1,800)
print(Y_hat)