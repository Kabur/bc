import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import gc
import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.utils import check_array

np.random.seed(7)
scaler = MinMaxScaler(feature_range=(0, 1))


def windowAndLabel(data, windowSize):
    X = []
    Y = []

    for i in range(len(data) - windowSize):
        X.append(data[i:(i + windowSize), 0])
        Y.append(data[i + windowSize, 0])
    return np.array(X), np.array(Y)


def load_data(filename, window_size, train_ratio):
    trainX = trainY = testX = testY = []

    dataframe = pandas.read_csv(filename, usecols=[2], engine='python')
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    datasetLength = len(dataset)

    dataset = scaler.fit_transform(dataset)

    # checkForZeros(dataset)

    trainSize = int(datasetLength * train_ratio)
    testSize = len(dataset) - trainSize

    trainSet = dataset[0:trainSize]
    testSet = dataset[trainSize:len(dataset)]

    trainX, trainY = windowAndLabel(trainSet, timesteps)
    testX, testY = windowAndLabel(testSet, timesteps)

    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

    return dataset, trainX, trainY, testX, testY


def createModel(trainX, trainY, epochs, batchSize=1):
    model = Sequential()
    # model.add(LSTM(4, batch_input_shape=(batchSize, timesteps, 1), stateful=True, return_sequences=True))
    model.add(LSTM(4, batch_input_shape=(batchSize, timesteps, 1), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())

    for _ in range(epochs):
        model.fit(trainX, trainY, epochs=1, batch_size=batchSize, verbose=2, shuffle=False)
        model.reset_states()

    print("Finished Training!")

    return model


def predictOneByOne(model, data, predictionLen):
    predictions = []

    for timestep in range(int(len(data)) - predictionLen):
        prediction = []
        for j in range(predictionLen):
            print(data[timestep][0][0])
            number = model.predict(data[timestep][0][0])  # needs a 3-dimensional array here
            prediction.append(number)

        predictions.append(prediction)

    return predictions


def predictMultiple(model, data, timesteps, predictionLength, batchSize, reset=0):
    result = []
    if reset == 0:
        reset = predictionLength * 4

    for i in range(int(len(data) / predictionLength)):
        # reset states
        if (i * predictionLength) >= reset:
            model.reset_states()

        curr_frame = data[i * predictionLength]
        predicted = []
        # predict 1 window for current window, then shift by windowSize(==timesteps)
        for j in range(predictionLength):
            prediction = model.predict(curr_frame[np.newaxis, :, :], batch_size=batchSize)
            predicted.append(prediction[0, 0])

            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [timesteps - 1], predicted[-1], axis=0)
        result.append(predicted)

    result = np.array(result)
    result = result.flatten()
    print("FLATTENED RESULT YO:")
    print(result)
    return result


def calculateMAPE(prediction, y):
    return np.mean(np.abs((y - prediction) / y)) * 100


if __name__ == '__main__':
    np.random.seed(7)
    epochs = 2
    timesteps = 3
    train_ratio = 0.67
    batchSize = 1
    day = 96
    week = day * 7

    # dataset, trainX, trainY, testX, testY = load_data('sample_energodata_aggregated.csv', timesteps, train_ratio)
    dataset, trainX, trainY, testX, testY = load_data('bigger_sample.csv', timesteps, train_ratio)
    # dataset, trainX, trainY, testX, testY = load_data('01_zilina_suma.csv', timesteps, train_ratio)
    model = createModel(trainX, trainY, epochs, batchSize)

    # predict
    testPredict = model.predict(testX, batch_size=batchSize)
    model.reset_states()
    testMultiple = predictMultiple(model, testX, timesteps, 3, batchSize)

    # invert predictions
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    testMultiple = scaler.inverse_transform([testMultiple])

    print("MULTIPLE PREDICTION:")
    print(testMultiple)
    print("testY:")
    print(testY)

    scoreSingle = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    print('scoreSingle: %.2f RMSE' % (scoreSingle))
    scoreSingleMAPE = calculateMAPE(testPredict[:, 0], testY[0])
    print("scoreSingleMAPE: %.2f MAPE" % (scoreSingleMAPE))
    scoreMultipleMAPE = calculateMAPE(testMultiple, testY[0])
    print("scoreMultipleMAPE: %.2f MAPE" % (scoreMultipleMAPE))


    gc.collect()
