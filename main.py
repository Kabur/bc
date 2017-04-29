
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


def load_data3(filename, timesteps, prediction_length, train_ratio, test_ratio, val_ratio):
    df = pd.read_csv(filename, usecols=[0, 2], parse_dates=[0], dayfirst=True, engine='python')
    # df = pd.read_csv(filename, usecols=[0, 3, 4, 5, 6, 7, 8, 9], parse_dates=[0], dayfirst=True, engine='python')

    ''' convert to float32 '''
    df['SUM_of_MNOZSTVO'] = df['SUM_of_MNOZSTVO'].values.astype('float32')

    ''' add day tags '''
    # df['DATUM'] = pd.to_datetime(df['DATUM'])
    df['day'] = df['DATUM'].dt.dayofweek
    df['DATUM'] = df['DATUM'].dt.date
    df['day'] = df.apply(tag_freedays, axis=1)
    del df['DATUM']

    ''' transform '''
    df['SUM_of_MNOZSTVO'] = scaler.fit_transform(df['SUM_of_MNOZSTVO'].values.reshape(-1, 1))

    dataset = df.values.astype('float32')

    ''' define sizes '''
    train_size = int(len(dataset) * (1 - val_ratio))

    ''' create sets '''
    train_set = dataset[0:train_size]
    val_set = dataset[train_size:len(dataset)]

    ''' label sets '''
    train_x, train_y = window_and_label(train_set, timesteps, prediction_length)
    val_x, val_y = window_and_label(val_set, timesteps, prediction_length)

    ''' shuffle '''
    train_x, train_y = shuffle_in_unison(train_x, train_y)

    return dataset, train_x, train_y, val_x, val_y


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
    dataset = dataset[0:int(len(dataset)*0.9)]
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
        self.test_losses.append(logs.get('val_loss'))
        val_loss = self.model.evaluate(self.val_x, self.val_y, self.batch_size, verbose=0)
        # self.test_losses.append(self.model.evaluate(self.test_x, self.test_y, self.batch_size, verbose=0))
        self.val_losses.append(val_loss)
        print('True val loss: ', val_loss)


def createModel2(model_name, train_x, train_y, val_x, val_y, epochs, timesteps, batch_size, prediction_length, features=1):
    model = Sequential()
    model.add(LSTM(48, input_shape=(timesteps, features), return_sequences=True))
    model.add(Dropout(0.15))
    model.add(LSTM(48, return_sequences=True))
    model.add(Dropout(0.15))
    model.add(LSTM(48, return_sequences=True))
    model.add(Dropout(0.15))
    model.add(LSTM(48))
    model.add(Dropout(0.15))
    model.add(Dense(prediction_length))
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())

    # filepath="E:\Dropbox\Bc\Python Projects\\bc_checkpoints\{epoch:02d}_checkpoint" + model_name + ".h5"
    # filepath="C:\Dropbox\Bc\Python Projects\\bc_checkpoints\{epoch:02d}_checkpoint" + model_name + ".h5"
    filepath="results\\bestcheckpoint" + model_name + ".h5"
    checkpoint = ModelCheckpoint(filepath, verbose=1, monitor='val_loss', save_best_only=True)

    history = LossHistory(model, batch_size, [], [], val_x, val_y)
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_split=0.15,
              verbose=2, shuffle=True, callbacks=[history, checkpoint])
    # model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_split=0.15,
    #           verbose=2, shuffle=True, callbacks=[history])

    batch_train_losses = np.array(history.batch_train_losses)
    train_losses = np.array(history.train_losses)
    test_losses = np.array(history.test_losses)
    val_losses = np.array(history.val_losses)

    print("Finished Training!")

    return model, batch_train_losses, train_losses, test_losses, val_losses


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

    filepath="E:\Dropbox\Bc\Python Projects\\bc_checkpoints\weights-improvement-{epoch:02d}.h5"
    checkpoint = ModelCheckpoint(filepath, verbose=1)

    history = LossHistory(model, batch_size, test_x, test_y, val_x, val_y)
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=2, shuffle=True, callbacks=[history, checkpoint])
    # model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=2, shuffle=True, callbacks=[history])

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
    timesteps = 96*7
    batch_size = 96*4
    prediction_length = 96
    # !!! UPDATE THIS BEFORE SAVING THE MODEL !!!
    epochs = 20
    total_epochs = 50
    model_name = '{0}model_branch6_lstm4_shape2_drop1_val5_days2'.format(total_epochs, batch_size, timesteps, features)
    # !!! UPDATE THIS BEFORE SAVING THE MODEL !!!

    ''' Load Data '''
    # dataset, train_x, train_y, test_x, test_y, val_x, val_y = load_data2('01_zilina_suma.csv', timesteps, prediction_length, train_ratio, test_ratio, val_ratio)
    dataset, train_x, train_y, val_x, val_y = load_data3('05_poprad_energo_cutend.csv', timesteps, prediction_length, train_ratio, test_ratio, val_ratio)
    print("Data Loaded!")

    ''' ***************************** OPTIONAL SECTION***************************************** '''
    ''' optional: Create Model'''
    # model, batch_train_losses, train_losses, test_losses, val_losses = createModel(train_x, train_y, test_x, test_y, val_x, val_y, epochs, timesteps, batch_size, prediction_length, features)
    # model, batch_train_losses, train_losses, test_losses, val_losses = createModel2(model_name, train_x, train_y, val_x, val_y, epochs, timesteps, batch_size, prediction_length, features)
    ''' optional: Load '''
    model = load_model('30model_branch6_lstm4_shape2_drop1_val5_days2.h5')
    print("Model Loaded!")
    # ''' optional: Return Training'''
    filepath="results\\bestcheckpoint" + model_name + ".h5"
    checkpoint = ModelCheckpoint(filepath, verbose=1, monitor='val_loss', save_best_only=True)
    history = LossHistory(model, batch_size, [], [], val_x, val_y)
    # model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=2, shuffle=True, callbacks=[history])
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_split=0.15,
              verbose=2, shuffle=True, callbacks=[history, checkpoint])
    batch_train_losses = np.array(history.batch_train_losses)
    train_losses = np.array(history.train_losses)
    test_losses = np.array(history.test_losses)
    val_losses = np.array(history.val_losses)
    ''' Save '''
    np.savetxt('results\\batch_train_losses' + model_name + '.txt', batch_train_losses, delimiter=',')
    np.savetxt('results\\train_losses' + model_name + '.txt', train_losses, delimiter=',')
    np.savetxt('results\\test_losses' + model_name + '.txt', test_losses, delimiter=',')
    np.savetxt('results\\val_losses' + model_name + '.txt', val_losses, delimiter=',')
    model.save('results\\' + model_name + '.h5', True)
    print("Model Saved!")

    ''' ***************************** OPTIONAL SECTION***************************************** '''
    ''' Predict '''
    prediction2 = model.predict(val_x, batch_size=batch_size)

    ''' Invert Predictions to RL values'''
    prediction2 = scaler.inverse_transform(prediction2)
    val_y = scaler.inverse_transform(val_y)

    ''' Calculate and print errors '''
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
    try:
        with open('results\\{0}model_and_resuklts.txt'.format(total_epochs), 'w') as file:
            file.write("prediction_vectors MAPE2: %.2f\n" % mape2)
            file.write("prediction_vectors Median Error2: %.2f\n" % median2)
            file.write("prediction_vectors Standard Deviation of Error2: %.2f\n" % standard_deviation2)
    except Exception as e:
        print("model_and_results didnt save, welp")
        print(e)

    ''' Plot Results'''
    plot_losses(train_losses, test_losses, val_losses)

    gc.collect()
