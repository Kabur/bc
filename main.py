
import os
import gc

import datetime
import numpy as np
# import pydot
import sys

from keras.callbacks import ModelCheckpoint
from matplotlib import style
import matplotlib.pyplot as plt
import pandas as pd
import math
import keras
from keras.layers.core import Dense, Dropout, Activation
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
holidays = [(2013, 1, 1), (2013, 1, 6), (2013, 3, 29), (2013, 4, 1), (2013, 5, 1),
            (2013, 5, 8), (2013, 7, 5), (2013, 8, 29), (2013, 9, 1), (2013, 9, 15),
            (2013, 11, 1), (2013, 11, 17), (2013, 12, 24), (2013, 12, 25), (2013, 12, 26),
            (2014, 1, 1), (2014, 1, 6), (2014, 4, 18), (2014, 4, 21), (2014, 5, 1),
            (2014, 5, 8), (2014, 7, 5), (2014, 8, 29), (2014, 9, 1), (2014, 9, 15),
            (2014, 11, 1), (2014, 11, 17), (2014, 12, 24), (2014, 12, 25), (2014, 12, 26),
            (2015, 1, 1), (2015, 1, 6), (2015, 4, 3), (2015, 4, 6), (2015, 5, 1),
            (2015, 5, 8), (2015, 7, 5), (2015, 8, 29), (2015, 9, 1), (2015, 9, 15),
            (2015, 11, 1), (2015, 11, 17), (2015, 12, 24), (2015, 12, 25), (2015, 12, 26)]


def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


def window_and_label(data, timesteps, prediction_length):
    x = []
    y = []

    for i in range(len(data) - (timesteps + prediction_length) + 1):
        x.append(data[i:(i + timesteps)])
        y.append(data[(i + timesteps):(i + timesteps + prediction_length), 0])
    return np.array(x), np.array(y)


def tag_freedays(row):
    if row['day'] is 5 or row['day'] is 6:
        return 1
    else:
        for date in holidays:
            if row['DATUM'] == datetime.date(*date):
                return 1

    return 0


def load_data2(filename, timesteps, prediction_length, train_ratio, test_ratio, val_ratio):
    df = pd.read_csv(filename, usecols=[0, 2], parse_dates=[0], dayfirst=True, engine='python')

    # df['SUM_of_MNOZSTVO'] = df['SUM_of_MNOZSTVO'].values.astype('float32')
    # df['DATUM'] = pd.to_datetime(df['DATUM'])
    df['day'] = df['DATUM'].dt.dayofweek
    df['DATUM'] = df['DATUM'].dt.date
    df['day'] = df.apply(tag_freedays, axis=1)

    del df['DATUM']
    df['SUM_of_MNOZSTVO'] = scaler.fit_transform(df['SUM_of_MNOZSTVO'].values.reshape(-1, 1))

    dataset = df.values.astype('float32')
    # dataset = scaler.fit_transform(dataset)

    train_size = int(len(dataset) * train_ratio)
    test_size = int(len(dataset) * test_ratio)
    val_size = int(len(dataset) * val_ratio)
    print(len(dataset), train_size, test_size)

    main_set = dataset[0:(train_size + test_size)]
    val_set = dataset[(train_size + test_size):len(dataset)]

    main_x, main_y = window_and_label(main_set, timesteps, prediction_length)
    val_x, val_y = window_and_label(val_set, timesteps, prediction_length)

    main_x, main_y = shuffle_in_unison(main_x, main_y)
    train_x = main_x[0: train_size]
    train_y = main_y[0: train_size]
    test_x = main_x[train_size:(train_size + test_size)]
    test_y = main_y[train_size:(train_size + test_size)]
    print("main_set", main_set.shape)
    print("*** main_x *** ", main_x.shape)
    print("*** main_y *** ", main_y.shape)
    print("*** train_x *** ", train_x.shape)
    print("*** train_y *** ", train_y.shape)
    print("*** test_x *** ", test_x.shape)
    print("*** test_y *** ", test_y.shape)

    return dataset, train_x, train_y, test_x, test_y, val_x, val_y


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
    # plt.savefig('graph.png')
    plt.close(fig)


def plot_losses(train_losses, test_losses, val_losses):
    fig = plt.figure()
    # ax = fig.add_subplot(111)
    plt.plot(train_losses, label='train_losses')
    plt.plot(test_losses, label='test_losses')
    plt.plot(val_losses, label='val_losses')
    plt.legend()
    plt.show()


class LossHistory(keras.callbacks.Callback):
    def __init__(self, model, batch_size, test_x, test_y, val_x, val_y):
        super().__init__()
        self.batch_train_losses = []
        self.train_losses = []
        self.test_losses = []
        self.val_losses = []
        self.model = model
        self.batch_size = batch_size
        self.test_x = test_x
        self.test_y = test_y
        self.val_x = val_x
        self.val_y = val_y

    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        self.batch_train_losses.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.train_losses.append(logs.get('loss'))
        self.test_losses.append(self.model.evaluate(self.test_x, self.test_y, self.batch_size, verbose=0))
        self.val_losses.append(self.model.evaluate(self.val_x, self.val_y, self.batch_size, verbose=0))


def createModel(train_x, train_y, test_x, test_y, val_x, val_y, epochs, timesteps, batch_size, prediction_length, features=1):
    model = Sequential()
    model.add(LSTM(48, input_shape=(timesteps, features), return_sequences=True))
    model.add(Dropout(0.15))
    model.add(LSTM(48, return_sequences=True))
    model.add(Dropout(0.15))
    model.add(LSTM(48, return_sequences=True))
    model.add(Dropout(0.15))
    # model.add(LSTM(48, return_sequences=True))
    # model.add(Dropout(0.15))
    model.add(LSTM(48))
    model.add(Dropout(0.15))
    model.add(Dense(prediction_length))
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())
    # plot_model(model, to_file='model.png')

    # filepath="E:\Dropbox\Bc\Python Projects\\bc_checkpoints\weights-improvement-{epoch:02d}.h5"
    # checkpoint = ModelCheckpoint(filepath, verbose=1)

    history = LossHistory(model, batch_size, test_x, test_y, val_x, val_y)
    # model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=2, shuffle=True, callbacks=[history, checkpoint])
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=2, shuffle=True, callbacks=[history])

    batch_train_losses = np.array(history.batch_train_losses)
    train_losses = np.array(history.train_losses)
    test_losses = np.array(history.test_losses)
    val_losses = np.array(history.val_losses)

    print("Finished Training!")

    return model, batch_train_losses, train_losses, test_losses, val_losses


if __name__ == '__main__':
    features = 2
    ''' Temp parameters'''
    # train_ratio = 0.50
    # test_ratio = 0.25
    # val_ratio = 0.25
    # epochs = 10
    # timesteps = 6
    # batch_size = 10
    # prediction_length = 3
    ''' True Parameters '''
    train_ratio = 0.70
    test_ratio = 0.15
    val_ratio = 0.15
    epochs = 2
    timesteps = 96*4
    batch_size = 96*4
    prediction_length = 96
    # !!! UPDATE THIS BEFORE SAVING THE MODEL !!!
    total_epochs = epochs
    # !!! UPDATE THIS BEFORE SAVING THE MODEL !!!

    ''' Load Data '''
    dataset, train_x, train_y, test_x, test_y, val_x, val_y = load_data2('01_zilina_suma.csv', timesteps, prediction_length, train_ratio, test_ratio, val_ratio)
    # dataset, train_x, train_y, test_x, test_y, val_x, val_y = load_data2('smaller_sample.csv', timesteps, prediction_length, train_ratio, test_ratio, val_ratio)
    print("Data Loaded!")

    ''' ***************************** OPTIONAL SECTION***************************************** '''
    ''' optional: Create Model'''
    model, batch_train_losses, train_losses, test_losses, val_losses = createModel(train_x, train_y, test_x, test_y, val_x, val_y, epochs, timesteps, batch_size, prediction_length, features)
    np.savetxt('{0}loss_history.txt'.format(total_epochs), batch_train_losses, delimiter=',')
    np.savetxt('{0}train_losses.txt'.format(total_epochs), train_losses, delimiter=',')
    np.savetxt('{0}test_losses.txt'.format(total_epochs), test_losses, delimiter=',')
    np.savetxt('{0}val_losses.txt'.format(total_epochs), val_losses, delimiter=',')
    ''' optional: Load '''
    # model = load_model('20e_model(48, 48, 48, 96)_shape(768, 384, 3).h5')
    # print("Model Loaded!")
    ''' optional: Return Training'''
    # history = LossHistory(model, batch_size, test_x, test_y, val_x, val_y)
    # batch_train_losses = np.array(history.batch_train_losses)
    # train_losses = np.array(history.train_losses)
    # test_losses = np.array(history.test_losses)
    # val_losses = np.array(history.val_losses)
    # model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=2, shuffle=True, callbacks=[history])
    # np.savetxt('{0}loss_history.txt'.format(total_epochs), batch_train_losses, delimiter=',')
    # np.savetxt('{0}train_losses.txt'.format(total_epochs), train_losses, delimiter=',')
    # np.savetxt('{0}test_losses.txt'.format(total_epochs), test_losses, delimiter=',')
    # np.savetxt('{0}val_losses.txt'.format(total_epochs), val_losses, delimiter=',')
    ''' Save '''
    model.save('{0}e_model(48, 48, 48, 48, 96)_shape({1}, {2}, {3})_drop1_val1_days2.h5'.format(total_epochs, batch_size, timesteps, features), True)
    print("Model Saved!")

    ''' ***************************** OPTIONAL SECTION***************************************** '''
    ''' Predict '''
    prediction = model.predict(test_x, batch_size=batch_size)
    prediction2 = model.predict(val_x, batch_size=batch_size)

    ''' Invert Predictions to RL values'''
    prediction = scaler.inverse_transform(prediction)
    prediction2 = scaler.inverse_transform(prediction2)
    test_y = scaler.inverse_transform(test_y)
    val_y = scaler.inverse_transform(val_y)

    ''' Calculate and print errors '''
    mape_per_vector = []
    for i in range(len(prediction)):
        mape_per_vector.append(calculate_mape(prediction[i], test_y[i]))
    mape_per_vector = np.array(mape_per_vector)

    mape = calculate_mape(prediction, test_y)
    median = np.median(mape_per_vector)
    standard_deviation = np.std(mape_per_vector)
    print("prediction_vectors MAPE: %.2f" % mape)
    print("prediction_vectors Median Error: %.2f" % median)
    print("prediction_vectors Standard Deviation of Error: %.2f" % standard_deviation)

    mape_per_vector2 = []
    for i in range(len(prediction2)):
        mape_per_vector2.append(calculate_mape(prediction2[i], val_y[i]))
    mape_per_vector2 = np.array(mape_per_vector2)

    mape2 = calculate_mape(prediction2, val_y)
    median2 = np.median(mape_per_vector2)
    standard_deviation2 = np.std(mape_per_vector2)
    print("prediction_vectors MAPE2: %.2f" % mape2)
    print("prediction_vectors Median Error2: %.2f" % median2)
    print("prediction_vectors Standard Deviation of Error2: %.2f" % standard_deviation2)

    ''' Plot Results'''  # saving fig to file doesnt work
    plot_losses(train_losses, test_losses, val_losses)
    # prediction_array = []
    # y_array = []
    # i = 0
    # while i < len(prediction):
    #     prediction_array.append(prediction2[i])
    #     y_array.append(val_y[i])
    #     i += prediction_length
    #
    # prediction_array = np.array(prediction_array).flatten()
    # y_array = np.array(y_array).flatten()
    # plot_results(prediction_array, y_array)

    gc.collect()
