import math
import random

import numpy as np


def sigmoid(xx):
    return 1 / (1 + math.exp(-xx))


def sigmoid_derivative(xx):

    xxx = math.exp(-xx)

    return xxx / ((1 + xxx)**2)


def error(target, actual):

    return ((target - actual)**2)/2


def error_derivative(target, actual):

    return target - actual


class InputLayer:

    def __init__(self, sz):
        self.size = sz

        self.preactivation_vector = np.zeros((sz,))
        self.activation_vector = np.zeros((sz,))
        self.partial_derivative_vector = np.zeros((sz,))

    def set_inputs(self, input_activation_vector):
        self.activation_vector = input_activation_vector


class HiddenLayer(InputLayer):

    def __init__(self, sz, pl, next_layer=None):

        self.next_layer = next_layer
        self.size = sz
        self.previous_layer = pl
        pl.next_layer = self

        self.preactivation_vector = np.zeros((sz,))
        self.activation_vector = np.zeros((sz,))
        self.bias_vector = np.zeros((sz,))
        self.error_vector = np.zeros((sz,))
        self.weight_matrix = np.zeros((sz, pl.size))



        height = self.weight_matrix.shape[0]
        width = self.weight_matrix.shape[1]

        for h in range(height):
            for w in range(width):
                self.weight_matrix[h, w] = random.random()

    # previous_layer is currently just one layer.  We might need to make this a list in order to accomomdate multiple previous layers
    def forward_propagate_layer(self):

        previous_activation_vector = self.previous_layer.activation_vector

        self.preactivation_vector = np.matmul(self.weight_matrix, previous_activation_vector)

        self.preactivation_vector += self.bias_vector

        for i, neuron in enumerate(self.preactivation_vector):
            self.activation_vector[i] = sigmoid(neuron)

    def set_error(self, target_vector):

        for i, target_value in enumerate(target_vector):

            self.error_vector[i] = error_derivative(target_value, self.activation_vector[i])  # de/da#
            self.error_vector[i] *= sigmoid_derivative(self.preactivation_vector[i])  # de/dp#

    def back_propagate_layer(self):

        pl = self.previous_layer

        pl.error_vector = np.matmul(self.error_vector, self.weight_matrix)  # de/da#

        for i, target_value in enumerate(pl.error_vector):
            pl.error_vector[i] *= sigmoid_derivative(pl.preactivation_vector[i])  # de/dp#

    def back_propagate_weights(self):

        pl = self.previous_layer

        shp = np.shape(self.weight_matrix)

        for xx in range(shp[0]):

            self.bias_vector[xx] += self.error_vector[xx]

            for yy in range(shp[1]):
                self.weight_matrix[xx, yy] += pl.activation_vector[yy] * self.error_vector[xx]








# first neural network with keras make predictions
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from google.colab import files
import pandas as pd
import io

# from google.colab import files
# uploaded = files.upload()

# load the dataset
dataset = pd.read_csv(io.BytesIO(uploaded['xor.data.csv']), header = -1)
X = dataset.iloc[:,0:2]
y = dataset.iloc[:,2]

# define the keras model
model = Sequential()
model.add(Dense(2, input_dim=2, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='mean_squared_error', optimizer="sgd", metrics=['accuracy'])

# fit the keras model on the dataset
model.fit(X, y, epochs=5000, batch_size=4, verbose=0)

# make class predictions with the model
predictions = model.predict_classes(X)

# summarize the first 5 cases
count = 0
for i in range(y.size):
  if predictions[i][0] == y[i]:
    count += 1

print("Accuracy = ", count/y.size)
