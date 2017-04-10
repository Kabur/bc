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


def window_and_label(data, window_size):
    x = []
    y = []

    for i in range(len(data) - window_size):
        x.append(data[i:(i + window_size), 0])
        y.append(data[i + window_size, 0])
    return np.array(x), np.array(y)


def load_data(filename, window_size, train_ratio):
    train_x = train_y = test_x = test_y = []

    dataframe = pandas.read_csv(filename, usecols=[2], engine='python')
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    dataset_length = len(dataset)

    dataset = scaler.fit_transform(dataset)

    train_size = int(dataset_length * train_ratio)
    testSize = len(dataset) - train_size

    train_set = dataset[0:train_size]
    test_set = dataset[train_size:len(dataset)]

    train_x, train_y = window_and_label(train_set, timesteps)
    test_x, test_y = window_and_label(test_set, timesteps)

    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))

    return dataset, train_x, train_y, test_x, test_y


def calculate_mape(prediction, y):
    return np.mean(np.abs((y - prediction) / y)) * 100


def plot_results(prediction, truth):
    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.plot(truth, label='True Data')
    plt.plot(prediction, label='Prediction')
    plt.legend()
    plt.show()


def createModel(train_x, train_y, epochs, batch_size):
    model = Sequential()
    # model.add(LSTM(4, batch_input_shape=(batch_size, timesteps, 1), stateful=True, return_sequences=True))
    model.add(LSTM(4, batch_input_shape=(batch_size, timesteps, 1), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())

    for _ in range(epochs):
        model.fit(train_x, train_y, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
        model.reset_states()

    print("Finished Training!")

    return model


def predictMultiple(model, data, timesteps, prediction_length, batch_size, reset=0):
    result = []
    if reset == 0:
        reset = prediction_length * 4

    for i in range(int(len(data) / prediction_length)):
        # reset states
        if (i * prediction_length) >= reset:
            model.reset_states() # todo: research how this works in Session bookmarks folder

        curr_frame = data[i * prediction_length]
        predicted = []
        # predict 1 window for current window, then shift by window_size(==timesteps)
        for j in range(prediction_length):
            prediction = model.predict(curr_frame[np.newaxis, :, :], batch_size=batch_size)
            predicted.append(prediction[0, 0])

            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [timesteps - 1], prediction[0, 0], axis=0)
        result.append(predicted)

    result = np.array(result)
    result = result.flatten()
    # print("multiple result")
    # print(result)
    return result


def predict_single(model, data, timesteps, batch_size=1):
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


if __name__ == '__main__':
    ''' Testing parameters'''
    train_ratio = 0.5
    epochs = 1
    timesteps = 96
    batch_size = 1
    prediction_length = 96
    ''' True parameteres'''
    # train_ratio = 0.70
    # epochs = 5
    # timesteps = 96
    # batch_size = 20
    # prediction_length = 96

    # dataset, train_x, train_y, test_x, test_y = load_data('01_zilina_suma.csv', timesteps, train_ratio)
    dataset, train_x, train_y, test_x, test_y = load_data('bigger_sample.csv', timesteps, train_ratio)
    ''' Shape Check'''
    # todo: make a check for train_x and prediction_length
        # maybe doesnt have to exit(), but rather predict what it can and then just stop.
    if dataset.shape[0] % batch_size is not 0:
        print("Number of samples must be divisible by batch_size!")
        exit()
    print("train_x.shape == ", train_x.shape)

    # model = createModel(train_x, train_y, epochs, batch_size)
    print("dataset len: ", len(dataset))
    print("train_x len: ", len(train_x))
    print("train_y len: ", len(train_y))
    print("test_x len: ", len(test_x))
    print("test_y len: ", len(test_y))
    exit()

    ''' Save & Load '''
    # model.save('model.h5', True)
    model = load_model('model_1_3_1_output1.h5')
    print("Model Loaded!")

    ''' Predict '''
    model.reset_states()
    prediction_single_keras = model.predict(test_x, batch_size=batch_size)
    model.reset_states()
    prediction_multiple = predictMultiple(model, test_x, timesteps, prediction_length, batch_size)
    model.reset_states()
    prediction_single = predict_single(model, test_x, timesteps, batch_size)

    ''' Invert Predictions to RL values'''
    prediction_single_keras = scaler.inverse_transform(prediction_single_keras)
    prediction_single = scaler.inverse_transform([prediction_single])
    test_y = scaler.inverse_transform([test_y])
    prediction_multiple = scaler.inverse_transform([prediction_multiple])
    print("Predictions done!")

    ''' Calculate MAPE and print'''
    mape_single = calculate_mape(prediction_single_keras[:, 0], test_y[0])
    mape_single2 = calculate_mape(prediction_single[:, 0], test_y[0])
    mape_multiple = calculate_mape(prediction_multiple[0], test_y[0])
    print("scoreSingleMAPE: %.2f MAPE" % (mape_single))
    print("scoreSingleMAPE2: %.2f MAPE" % (mape_single2))
    print("scoreMultipleMAPE: %.2f MAPE" % (mape_multiple))

    ''' Plot Results'''
    plot_results(prediction_multiple[0], test_y[0])

    gc.collect()
