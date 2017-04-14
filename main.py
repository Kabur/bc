import os
import gc
import numpy as np
# import pydot
import sys
from matplotlib import style
import matplotlib.pyplot as plt
import pandas as pd
import math
import keras
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.models import load_model
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.utils import check_array

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
np.random.seed(7)
scaler = MinMaxScaler(feature_range=(-1, 1))


def window_and_label(data, timesteps, prediction_length):
    x = []
    y = []

    for i in range(len(data) - (timesteps + prediction_length) + 1):
        x.append(data[i:(i + timesteps)])
        y.append(data[(i + timesteps):(i + timesteps + prediction_length), 0])
    return np.array(x), np.array(y)


def load_data2(filename, timesteps, prediction_length, train_ratio, test_ratio, validation_ratio):
    df = pd.read_csv(filename, usecols=[0, 2], engine='python')

    # df['SUM_of_MNOZSTVO'] = df['SUM_of_MNOZSTVO'].values.astype('float32')
    df['DATUM'] = pd.to_datetime(df['DATUM'])
    df['day'] = df['DATUM'].dt.dayofweek
    df['day_sine'] = df['day'].apply(np.sin)
    df['day_cosine'] = df['day'].apply(np.cos)
    del df['DATUM']
    del df['day']
    df['SUM_of_MNOZSTVO'] = scaler.fit_transform(df['SUM_of_MNOZSTVO'].values.reshape(-1, 1))

    dataset = df.values.astype('float32')
    # dataset = scaler.fit_transform(dataset)

    train_size = int(len(dataset) * train_ratio)
    test_size = int(len(dataset) * test_ratio)

    # todo: shuffle together train and test data here

    train_set = dataset[0:train_size]
    test_set = dataset[train_size:train_size + test_size]
    validation_set = dataset[train_size + test_size:len(dataset)]

    train_x, train_y = window_and_label(train_set, timesteps, prediction_length)
    test_x, test_y = window_and_label(test_set, timesteps, prediction_length)
    validation_x, validation_y = window_and_label(validation_set, timesteps, prediction_length)

    return dataset, train_x, train_y, test_x, test_y, validation_x, validation_y


def shape_check(train_x, train_y, test_x, test_y, batch_size, prediction_length):
    ''' Shape Check'''
    if len(train_x) % batch_size is not 0:
        print("Number of training samples is not divisible by batch_size")
        exit()
    if len(test_x) % batch_size is not 0:
        print("Number of testing samples is not divisible by batch_size")
        exit()
    if len(test_y) % prediction_length is not 0:
        print("Number of samples is not divisible by prediction_length, creating test_y2 for sequence prediction")
        cutoff = len(test_y) % prediction_length
        print("len(test_y): ", len(test_y), ",prediction_length: ", prediction_length, ",cutoff: ", cutoff)
        test_y = test_y[:len(test_y) - cutoff]
        print("new len(test_y): ", len(test_y))

    return test_y


def calculate_mape(prediction, y):
    return np.mean(np.abs((y - prediction) / y)) * 100


def plot_results(prediction, truth):
    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.plot(truth, label='True Data')
    plt.plot(prediction, label='Prediction')
    plt.legend()
    plt.show()
    plt.savefig('graph.png')
    plt.close(fig)


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def createModel(train_x, train_y, epochs, timesteps, batch_size, prediction_length, features=1):
    model = Sequential()
    model.add(LSTM(10, input_shape=(timesteps, features), return_sequences=True))
    model.add(LSTM(10, return_sequences=True))
    model.add(LSTM(10, return_sequences=True))
    model.add(LSTM(10))
    model.add(Dense(prediction_length))
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())
    # plot_model(model, to_file='model.png')


    history = LossHistory()
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=2, shuffle=True, callbacks=[history])

    print("Finished Training!")

    return model, np.array(history.losses)


def predict_sequence(model, data, timesteps, batch_size, prediction_length, reset=0):
    result = []

    for i in range(int(len(data) / prediction_length)):
        curr_frame = data[i * prediction_length]
        predicted = []
        # predict 1 window for current window, then shift by window_size(==timesteps)
        for j in range(prediction_length):
            prediction = model.predict(curr_frame[np.newaxis, :, :], batch_size=batch_size)
            predicted.append(prediction[0, 0])

            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [timesteps - 1], prediction[0, 0], axis=0)

        # reset the state after 'reset' sequence predictions
        if reset != 0 and (i + 1) % reset == 0:
            model.reset_states()

        # append the predicted sequence
        result.append(predicted)

    result = np.array(result)
    result = result.flatten()
    return result


def predict_single(model, data, batch_size, reset=0):
    result = []

    for i, sample in enumerate(data):
        prediction = model.predict(sample[np.newaxis, :, :], batch_size=batch_size)
        if reset != 0 and (i + 1) % reset == 0:
            model.reset_states()
        result.append(prediction)

    result = np.array(result).flatten()
    return result


if __name__ == '__main__':
    features = 3
    ''' Temp parameters'''
    train_ratio = 0.50
    test_ratio = 0.25
    validation_ratio = 0.25
    epochs = 3
    timesteps = 6
    batch_size = 10
    prediction_length = 3
    ''' True Parameters '''  # 9.43 MAPE
    # train_ratio = 0.70
    # test_ratio = 0.20
    # validation_ratio = 0.10
    # epochs = 10
    # timesteps = 96*4
    # batch_size = 96*4
    # prediction_length = 96
    # todo: tweak parameters

    # dataset, train_x, train_y, test_x, test_y, validation_x, validation_y = load_data2('01_zilina_suma.csv', timesteps, prediction_length, train_ratio, test_ratio, validation_ratio)
    dataset, train_x, train_y, test_x, test_y, validation_x, validation_y = load_data2('bigger_sample.csv', timesteps, prediction_length, train_ratio, test_ratio, validation_ratio)
    print("Data Loaded!")

    model, loss_history = createModel(train_x, train_y, epochs, timesteps, batch_size, prediction_length, features)

    ''' Load & Save '''
    # model = load_model('model(10, 10, 10, 96)_shape(384, 384, 3).h5')
    # print("Model Loaded!")
    # history = LossHistory()
    # model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=2, shuffle=True, callbacks=[history])
    # np.savetxt('loss_history.txt', np.array(history.losses), delimiter=',')
    # model.save('model(10, 10, 10, 96)_shape(384, 384, 3).h5', True)
    # print("Model Saved!")

    ''' Predict '''
    prediction = model.predict(test_x, batch_size=batch_size)

    ''' Invert Predictions to RL values'''
    prediction = scaler.inverse_transform(prediction)
    test_y = scaler.inverse_transform(test_y)

    ''' Calculate and print errors '''
    mape_per_vector = []
    for i in range(len(prediction)):
        mape_per_vector.append(calculate_mape(prediction[i], test_y[i]))
    mape_per_vector = np.array(mape_per_vector)

    mape = calculate_mape(prediction, test_y)
    median = np.median(mape_per_vector)
    standard_deviation = np.std(mape_per_vector)
    print("prediction_vectors MAPE: %.2f" % mape)
    print("prediction_vectors Median: %.2f" % median)
    print("prediction_vectors Standard Deviation: %.2f" % standard_deviation)

    ''' Plot Results'''  # saving fig to file doesnt work
    prediction_array = []
    y_array = []
    i = 0
    while i < len(prediction):
        prediction_array.append(prediction[i])
        y_array.append(test_y[i])
        i += prediction_length

    print("******** PREDICTION ARRAY ********* ")
    print(np.array(prediction_array))
    prediction_array = np.array(prediction_array).flatten()
    y_array = np.array(y_array).flatten()
    print("******** PREDICTION ARRAY ********* ")
    print(prediction_array)
    plot_results(prediction_array, y_array)

    ''' SAVE RESULTS AND THE MODEL SUMMARY TO FILES '''
    np.savetxt('loss_history.txt', loss_history, delimiter=',')
    orig_stdout = sys.stdout
    file = open('model_summary.txt', 'w')
    sys.stdout = file
    print(model.summary())
    print('epochs: ', epochs)
    print('timesteps: ', timesteps)
    print('batch_size: ', batch_size)
    print('prediction_length: ', prediction_length)
    print('features: ', features)
    print('mape: ', mape)
    print('median: ', median)
    print('standard_deviation: ', standard_deviation)
    sys.stdout = orig_stdout
    file.close()

    gc.collect()
