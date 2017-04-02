import gc
import numpy as np
import pandas
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()


def load_data(filename, window_size, train_ratio):
    f = open(filename, 'rb').read()
    data = f.decode().split('\n')

    # print("Data: ")
    # print(data)

    # window_size = 3
    sequence_length = window_size + 1

    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    # print("Result:")
    # print(result)

    normalised_data = []
    for window in result:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)

    result = normalised_data
    result = np.array(result)
    # print("Normalized: ")
    # print(result)

    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    # print("After train = result[:int(row), :]")
    # print(train)

    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    # print("After shuffle and x_train = train[:, :-1]")
    # print(x_train)
    # print("After shuffle and y_train = train[:, -1]")
    # print(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # print("x_train after reshape")
    # print(x_train)
    print("x_train")
    print(x_train)
    print("y_train")
    print(y_train)
    print("x_test")
    print(x_test)
    print("y_test")
    print(y_test)

    return [x_train, y_train, x_test, y_test]


def build_model(layers):
    model = Sequential()

    model.add(LSTM(input_dim=layers[0], output_dim=layers[1], return_sequences=True))
    model.add(LSTM(layers[2], return_sequences=False))
    model.add(Dense(output_dim=layers[3]))
    model.add(Activation("linear"))

    model.compile(loss="mse", optimizer="rmsprop", metrics=["accuracy"])
    # try optimizer adam

    print("Model compiled")
    print(model.summary())
    return model


# predict sequence of 50 steps before shifting prediction run forward by 50 steps
def predict_sequences_multiple(model, data, window_size, prediction_len):
    prediction_seqs = []
    for i in range(int(len(data) / prediction_len)):
        curr_frame = data[i * prediction_len]
        # print("************* curr_frame *************")
        # print(curr_frame)

        predicted = []
        for j in range(prediction_len):
            print("curr_frame:")
            print(curr_frame[np.newaxis, :, :])
            exit()
            keepo = model.predict(curr_frame[np.newaxis, :, :])
            kappa = keepo[0, 0]
            # print("???")
            # print(curr_frame[np.newaxis, :, :])
            # print("keepo: ")
            # print(keepo)
            # print("???")

            predicted.append(kappa)
            # print("Kappa:")
            # print(kappa)

            curr_frame = curr_frame[1:]
            # print("curr_frame:")
            # print(curr_frame)

            curr_frame = np.insert(curr_frame, [window_size - 1], predicted[-1], axis=0)
            # print(curr_frame)

        # print("Predicted: ")
        # print(predicted)
        prediction_seqs.append(predicted)
    return prediction_seqs


if __name__ == '__main__':
    epochs = 1
    window_size = 3
    prediction_len = 3

    # Load the data
    # trainX, trainY, testX, testY = load_data('smaller_sample.csv', window_size, 0.66)
    trainX, trainY, testX, testY = load_data('sinewave_sample.csv', window_size, 0.66)
    # trainX, trainY, testX, testY = load_data('sp500.csv', window_size, 0.66)

    print("Data loaded!")

    # Build the model
    model = build_model([1, 50, 100, 1])

    model.fit(trainX, trainY, batch_size=512, nb_epoch=epochs)
    print("Training Finished!")
    predictions = predict_sequences_multiple(model, testX, window_size, prediction_len)

    # scores = model.evaluate(testX, testY, verbose=0)
    # print("Accuracy: %.2f%%" % (scores[1]*100))

    # print("Predictions: ")
    # print(predictions)
    # print("The truth:")
    # print(testY)

    # plot_results_multiple(predictions, testY, prediction_len)
    print("done")

