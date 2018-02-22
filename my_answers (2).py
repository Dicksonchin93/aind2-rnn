import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dense, Activation, LSTM

import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs   
    X = []
    y = []
    i = 0
    for id, data in enumerate(series):
        if(i>=window_size):
            y.append([data])
            temp = []
            for win in range(window_size,0,-1):
                temp.append(series[id-win])
            X.append(temp)
        i+=1
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(5, 1)))
    model.add(Dense(1))
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']

    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    track_id = 2
    inputs = []
    outputs = []
    size = len(text)
    step = 2
    for id, data in enumerate(text):
        if (step==step_size):
            if(id >= window_size):
                temp = []
                for win in range(window_size,0,-1):
                    temp.append(text[id-win])
                inputs.append(temp)
                outputs.append(text[id])
            step = 0
        step += 1


    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(100, num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation('softmax'))
    return model
