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


def load_data2(filename, timesteps, train_ratio):
    train_x = train_y = test_x = test_y = []

    df = pandas.read_csv(filename, usecols=[0], engine='python')
    print(df)

    df['DATUM'] = pandas.to_datetime(df['DATUM'])
    df['day_of_week'] = df['DATUM'].dt.weekday_name

    print(df)
    exit()

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


# bigger the batch_size, the better the GPU performs
def createModel(train_x, train_y, epochs, batch_size, features=1):
    model = Sequential()
    model.add(LSTM(4, batch_input_shape=(batch_size, timesteps, 1), stateful=True, return_sequences=True))
    model.add(LSTM(4, batch_input_shape=(batch_size, timesteps, features), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())

    for _ in range(epochs):
        model.fit(train_x, train_y, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
        model.reset_states()

    print("Finished Training!")

    return model


# a stateful model has to be trained and tested on the same batch_size
#
# def predict_single_test(model, data):
#     result = []
#     predicted = []
#     for i in range(int(len(data) / 2)):
#         curr_frame = data[i*2: i*2+2]
#         prediction = model.predict(curr_frame, batch_size=2)
#         # model.reset_states()
#         print(curr_frame)
#         print(prediction)
#         result.append(prediction)
#
#     print("RESULT")
#     print(np.array(result).flatten())
#     exit()
#     return np.array(result).flatten()


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
    features = 2
    train_ratio = 0.50
    ''' Temp parameters'''
    epochs = 1
    timesteps = 3
    batch_size = 1
    prediction_length = 3
    reset = 0
    ''' True parameteres'''
    # epochs = 5
    # timesteps = 96
    # batch_size = 1
    # prediction_length = 96
    # reset = 96  # 96 * 7
    # todo: tweak parameters
    # todo: increase batch_size, compute on GPU

    # dataset, train_x, train_y, test_x, test_y = load_data('01_zilina_suma.csv', timesteps, train_ratio)
    # dataset, train_x, train_y, test_x, test_y = load_data('bigger_sample.csv', timesteps, train_ratio)
    dataset, train_x, train_y, test_x, test_y = load_data2('smaller_sample.csv', timesteps, train_ratio)
    print("dataset len: ", len(dataset))
    print("train_x len: ", len(train_x))
    print("train_y len: ", len(train_y))
    print("test_x len: ", len(test_x))
    print("test_y len: ", len(test_y))
    test_y2 = shape_check(test_x, train_y, test_x, test_y, batch_size, prediction_length)

    model = createModel(train_x, train_y, epochs, batch_size, features)

    ''' Save & Load '''
    model.save('model_1_96_1_output1.h5', False)
    print("Model Saved!")
    # model = load_model('model_1_3_1_output1.h5')
    # print("Model Loaded!")

    ''' Predict '''
    model.reset_states()
    prediction_single_keras = model.predict(test_x, batch_size=batch_size)
    model.reset_states()
    prediction_single = predict_single(model, test_x, batch_size, reset=reset)
    model.reset_states()
    prediction_sequence = predict_sequence(model, test_x, timesteps, batch_size, prediction_length, reset=reset)

    ''' Invert Predictions to RL values'''
    prediction_single_keras = scaler.inverse_transform(prediction_single_keras)
    prediction_single = scaler.inverse_transform([prediction_single])
    test_y = scaler.inverse_transform([test_y])
    test_y2 = scaler.inverse_transform([test_y2])
    prediction_sequence = scaler.inverse_transform([prediction_sequence])
    print("Predictions done!")

    ''' Calculate MAPE and print'''
    # try:
    mape_single_keras = calculate_mape(prediction_single_keras[:, 0], test_y[0])
    mape_single = calculate_mape(prediction_single[0], test_y[0])
    mape_sequence = calculate_mape(prediction_sequence[0], test_y2[0])
    # except ValueError as e:
    #     print("Number of samples must be divisible by prediction_length")
    #     exit()
    print("mape_single_keras: %.2f MAPE" % mape_single_keras)
    print("mape_single: %.2f MAPE" % mape_single)
    print("mape_sequence: %.2f MAPE" % mape_sequence)

    ''' Plot Results'''
    plot_results(prediction_sequence[0], test_y2[0])

    gc.collect()
