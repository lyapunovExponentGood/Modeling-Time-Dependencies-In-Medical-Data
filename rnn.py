import tensorflow as tf
import numpy as np
import matplotlib.pyplot as matplot
import math as mth

from matplotlib import pyplot as plt
from tensorflow import keras
from keras import layers, models, optimizers
from keras.models import Sequential
from keras.layers import Dense
from datetime import datetime

'''
#The following reads in code from the raw data file:

import numpy as np
import csv

file = open('SeizuretrackerSample.csv')
csvreader = csv.reader(file)

rows = []

#get through initial headers 
for i in range(0,25):
  nextRow = next(csvreader) 

for i in range(0, 2639-25):
  nextRow = next(csvreader)
  rows.append(nextRow)

'''

# the following code reads in the InterseizureInterval csv file
# ISIs are already parsed so no need for the converToDateTimes or

import csv

file = open('InterSeizureInterval.csv')
csvreader = csv.reader(file)
ISI = np.zeros((2613, 1))

for i in range(0, 2613):
    nextRow = next(csvreader)
    ISI[i] = float(nextRow[0])
ISI = ISI[:2294]


def convertToDateTimes(list):
    dateList = []

    i = 0
    for row in list:
        date = row[1]
        dateTime = datetime.strptime(date, '%Y-%m-%d %X')
        dateList.append(dateTime)

    return dateList


def ISICalc(dateList):
    ISI = np.zeros((len(dateList) - 1, 1))

    for i in range(0, len(dateList) - 1):
        delta = dateList[i + 1] - dateList[i]
        isi = delta.total_seconds()
        ISI[i] = isi / (60 * 60 * 24)  # express ISIs in days
    return ISI


def TimeEmbedding(array, d, n):
    m = mth.floor(n / (d + 1))  # number of embeddings + correct one forecast into the future predictions
    embeddings = np.zeros([m, d])
    answers = np.zeros([m, 1])

    indx = 0
    for i in range(0, m):
        embeddings[i, 0:d] = array[0, indx:indx + d]
        # print('index: ',i)
        answers[i, 0] = array[0, (indx + d)]
        indx = indx + d + 1

    return [embeddings, answers, d, m]


'''
#this code generates uniform random data between the min and max of the normal ISI training data matrix: 
import numpy as np
import numpy.random as random


ISI = np.zeros((2613, 1))

for i in range(0, 2613):
  ISI[i] = random.uniform(low=0.0, high=7.0625, size=None)

'''
n = len(ISI)
# ISIt = np.transpose(ISI)
proportionTraining = 3 / 4  # This parameter sets what percentage of our data will be used to train the network
trainArray = np.transpose(ISI[0:mth.floor(n * proportionTraining)])

[embeddingsX, answersX, dX, mX] = TimeEmbedding(trainArray, 3, 3 * n / 4)  # four dimensions

testArray = np.transpose(ISI[mth.floor(3*n/4):n])
[embeddingsTestX, answersTestX, dTest, mTestX] = TimeEmbedding(testArray, 3, mth.floor(n/4))

inputs = np.concatenate([embeddingsX, ], axis=1)
correctOutputs = np.concatenate([answersX, ], axis=1)

inputsTest = np.concatenate([embeddingsTestX, ], axis = 1)
correctTestOutputs = np.concatenate([answersTestX, ], axis = 1)

function_approximater_A = Sequential()

function_approximater_A.add(layers.Embedding(input_dim=1000, output_dim=64))
# The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
function_approximater_A.add(layers.GRU(256, return_sequences=True))
# The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
function_approximater_A.add(layers.SimpleRNN(128))
function_approximater_A.add(layers.Dense(10))


# #function_approximater_A.add(Dense(units=3, activation='linear', input_shape=(3,)))
# function_approximater_A.add(Dense(units = 2048, activation = 'sigmoid', input_dim = 1))
# function_approximater_A.add(Dense(units = 1024, activation = 'sigmoid', input_dim = 1))
# function_approximater_A.add(Dense(units = 512, activation = 'sigmoid', input_dim = 1))
# #function_approximater_A.add(Dense(units=256, activation='sigmoid', input_dim=1))
# #function_approximater_A.add(Dense(units=128, activation='sigmoid', input_dim=1))
# #function_approximater_A.add(Dense(units=64, activation='sigmoid', input_dim=1))
# #function_approximater_A.add(Dense(units=32, activation='sigmoid', input_dim=1))
# #function_approximater_A.add(Dense(units=16, activation='sigmoid', input_dim=1))
# #function_approximater_A.add(Dense(units=8, activation='sigmoid', input_dim=1))
# function_approximater_A.add(Dense(units = 4, activation = 'sigmoid', input_dim = 1))
# #function_approximater_A.add(Dense(units=1, activation='linear', input_dim=1))

# sgd_1 = tf.keras.optimizers.SGD(learning_rate=0.001)
function_approximater_A.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')

# running this trains the network

# Batch size equal to dataset length - all data is used during backpropagation (classic gradient descent):
# function_approximater_A.fit(inputs, correctOutputs, batch_size = len(answersX), epochs = 20)

# Batch size equal to dataset length - only one datapoint is used during backpropagation (stochastic gradient descent):
function_approximater_A.fit(inputs, correctOutputs, batch_size=8, epochs=50)

y_pred = function_approximater_A.predict(inputsTest)
print(y_pred)
plt.plot(y_pred)
plt.plot(correctTestOutputs)
plt.show()

index = np.shape(correctTestOutputs)
m = index[0]
y_error = (1/(m)*np.sum(np.square(y_pred - correctTestOutputs), axis=0))
#error_1 = 0

print(y_error) 