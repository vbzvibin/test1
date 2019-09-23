# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 14:18:03 2019

@author: vibinan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.backend import clear_session




import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  

from keras.models import Sequential  
from keras.layers import Dense  
from keras.layers import LSTM  
from keras.layers import Dropout 

## Import Dataset

# Dataset link - https://finance.yahoo.com/quote/AAPL/history?p=AAPL&.tsrc=fin-srch

apple_training_complete = pd.read_csv(r'E:\Datasets\apple_training.csv')  

apple_training_processed = apple_training_complete.iloc[:, 1:2].values  


## Data Normalization

from sklearn.preprocessing import MinMaxScaler  
scaler = MinMaxScaler(feature_range = (0, 1))

apple_training_scaled = scaler.fit_transform(apple_training_processed)

## Convert Training Data to Right Shape

features_set = []  
labels = []  
for i in range(60, 1260):  
    features_set.append(apple_training_scaled[i-60:i, 0])
    labels.append(apple_training_scaled[i, 0])
    
features_set, labels = np.array(features_set), np.array(labels)  

features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1)) 

### Training The LSTM

model = Sequential() 

model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], 1)))  

model.add(Dropout(0.2)) 

model.add(LSTM(units=50, return_sequences=True))  
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))  
model.add(Dropout(0.2))

model.add(LSTM(units=50))  
model.add(Dropout(0.2)) 

### Creating Dense Layer

model.add(Dense(units = 1))

### Model Compilation

model.compile(optimizer = 'adam', loss = 'mean_squared_error')  

### Algorithm Training

model.fit(features_set, labels, epochs = 100, batch_size = 32)  

### Testing our LSTM

apple_testing_complete = pd.read_csv(r'E:\Datasets\apple_testing.csv')  
apple_testing_processed = apple_testing_complete.iloc[:, 1:2].values

### Converting Test Data to Right Format

apple_total = pd.concat((apple_training_complete['Open'], apple_testing_complete['Open']), axis=0)  

test_inputs = apple_total[len(apple_total) - len(apple_testing_complete) - 60:].values  


test_inputs = test_inputs.reshape(-1,1)  
test_inputs = scaler.transform(test_inputs) 


test_features = []  
for i in range(60, 80):  
    test_features.append(test_inputs[i-60:i, 0])
    
test_features = np.array(test_features)  
test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))  

## Making Predictions

predictions = model.predict(test_features) 

predictions = scaler.inverse_transform(predictions)  


plt.figure(figsize=(10,6))  
plt.plot(apple_testing_processed, color='blue', label='Actual Apple Stock Price')  
plt.plot(predictions , color='red', label='Predicted Apple Stock Price')  
plt.title('Apple Stock Price Prediction')  
plt.xlabel('Date')  
plt.ylabel('Apple Stock Price')  
plt.legend()  
plt.show()      

































