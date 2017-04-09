import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import gc
import numpy as np
from matplotlib import style
import matplotlib.pyplot as plt
import pandas
import math
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.models import load_model
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
    model.add(LSTM(4, batch_input_shape=(batchSize, timesteps, 1), stateful=True, return_sequences=True))
    model.add(LSTM(4, batch_input_shape=(batchSize, timesteps, 1), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())

    for _ in range(epochs):
        model.fit(trainX, trainY, epochs=1, batch_size=batchSize, verbose=2, shuffle=False)
        model.reset_states()

    print("Finished Training!")

    return model


# def predictOneByOne(model, data, predictionLen):
#     predictions = []
#
#     for timestep in range(int(len(data)) - predictionLen):
#         prediction = []
#         for j in range(predictionLen):
#             print(data[timestep][0][0])
#             number = model.predict(data[timestep][0][0])  # needs a 3-dimensional array here
#             prediction.append(number)
#
#         predictions.append(prediction)
#
#     return predictions


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
    # print("multiple result")
    # print(result)
    return result


def predictSingle(model, data, timesteps, batch_size=1):
    # states are reset automatically after batch_size
    result = []
    print(data.shape)
    for i in range(int(len(data))):
        curr_window = data[i]
        prediction = model.predict(curr_window[np.newaxis, :, :], batch_size=batch_size)
        result.append(prediction)

    result = np.array(result).flatten()
    # print("single result")
    # print(result)
    return result


def calculateMAPE(prediction, y):
    return np.mean(np.abs((y - prediction) / y)) * 100


def plot_results(prediction, truth):
    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.plot(truth, label='True Data')
    plt.plot(prediction, label='Prediction')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    np.random.seed(7)
    train_ratio = 0.67
    day = 96
    week = day * 7
    # testing parameters
    epochs = 1
    timesteps = 3
    batchSize = 1
    # true parameters
    # epochs = 5
    # timesteps = day
    # batchSize =

    # dataset, trainX, trainY, testX, testY = load_data('smaller_sample.csv', timesteps, train_ratio)
    dataset, trainX, trainY, testX, testY = load_data('bigger_sample.csv', timesteps, train_ratio)
    # dataset, trainX, trainY, testX, testY = load_data('01_zilina_suma.csv', timesteps, train_ratio)
    # model = createModel(trainX, trainY, epochs, batchSize)
    print(testX)
    print("**********")
    print(testY)

    ''' SAVE & LOAD '''
    # model.save('model.h5', True)
    model = load_model('model.h5')
    print("Model Loaded!")

    # predict
    model.reset_states()
    prediction_single_keras = model.predict(testX, batch_size=batchSize)
    model.reset_states()
    prediction_multiple = predictMultiple(model, testX, timesteps, 3, batchSize)
    model.reset_states()
    prediction_single = predictSingle(model, testX, timesteps, batchSize)

    # invert predictions
    prediction_single_keras = scaler.inverse_transform(prediction_single_keras)
    prediction_single = scaler.inverse_transform([prediction_single])
    testY = scaler.inverse_transform([testY])
    prediction_multiple = scaler.inverse_transform([prediction_multiple])
    print("Predictions done!")
    # print("TEST PREDICT:")
    # print(testPredict)
    # print("MULTIPLE PREDICTION:")
    # print(testMultiple)
    # print("testY:")
    # print(testY)
    # print("testPredict[:, 0]")
    # print(testPredict[:, 0])

    # print("testPredict len: ", len(testPredict))
    # print("testMultiple[0] len: ", len(testMultiple[0]))
    # print("testY[0] len: ", len(testY[0]))

    print("single keras..", len(prediction_single_keras))
    print(prediction_single_keras)
    print("single custom..", len(prediction_single))
    print(prediction_single)
    print("multiple..", len(prediction_multiple))
    print(prediction_multiple)
    print("testY.. ", len(testY))
    print(testY)
    exit()

    scoreSingle = math.sqrt(mean_squared_error(testY[0], prediction_single_keras[:, 0]))
    print('scoreSingle: %.2f RMSE' % (scoreSingle))
    scoreSingleMAPE = calculateMAPE(prediction_single_keras[:, 0], testY[0])
    print("scoreSingleMAPE: %.2f MAPE" % (scoreSingleMAPE))
    scoreSingleMAPE2 = calculateMAPE(prediction_single[:, 0], testY[0])
    print("scoreSingleMAPE2: %.2f MAPE" % (scoreSingleMAPE2))

    try:
        scoreMultipleMAPE = calculateMAPE(np.append(prediction_multiple[0], testY[0][-1]), testY[0])
        print("scoreMultipleMAPE: %.2f MAPE" % (scoreMultipleMAPE))
        plot_results(np.append(prediction_multiple[0], testY[0][-1]), testY[0])
    except ValueError as e:
        print("didnt go through")
        print(e)

    gc.collect()
