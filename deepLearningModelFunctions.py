# import keras
# from keras.models import Sequential
# from keras.layers import Dense, LSTM


# def LSTMModel(n_input, n_features, neurons, epochs, activationFunction, optimizer, lossFunction):

#     # Define Model
#     LSTMmodel = Sequential()
#     LSTMmodel.add(LSTM(neurons, activation=activationFunction,
#                        input_shape=(n_input, n_features)))
#     LSTMmodel.add(Dense(1))  # Output Layer
#     LSTMmodel.compile(optimizer=optimizer, loss=lossFunction)

#     return LSTMmodel
