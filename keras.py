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
