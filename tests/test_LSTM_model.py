"""
Tests for LSTMNetwork
"""

import anomaly_detection.time_series_generator as gen
import math
import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


"""
Helper function
"""
# range is a tuple with (start, end) for time variable
def generate_trig_series(dim, range):

    dim1_func = math.sin
    dim2_func = math.cos

    def dim3_func(x):
        return 0.5*math.sin(x)

    if (dim == 1):
        functions = [dim1_func]
    elif (dim == 2):
        functions = [dim1_func, dim2_func]
    elif (dim == 3):
        functions = [dim1_func, dim2_func, dim3_func]
    else:
        assert(False)

    series = gen.generate_time_series(dim=dim, t_range=range,
                                      count=1000, functions=functions, is_anomolous=0)

    return series


def plot_series(series, title):

    dim = len(series[0].X)
    t = [point.t for point in series]

    for d in range(dim):
        x = [point.X[d] for point in series]
        plt.plot(t, x, label=title + '_dimension-' + str(d))

    plt.title(title)
    plt.xlabel("t")
    plt.ylabel("y")
    plt.legend(loc='upper right')


# Given a series with multiple timesteps, extract the series for each timestep
# Return those multiple series in a tuple
def seperate_multi_timestep_series(mult_series, dim, timesteps):
    dataset_size = len(mult_series)
    ret_mult_series = (np.zeros((dataset_size, dim)),) * timesteps

    n = 0
    for sample in mult_series:
        for step in range(timesteps):
            curr_series = ret_mult_series[step]
            for d in range(dim):
                curr_series[n][d] = sample[step+d]

        n += 1

    return ret_mult_series


###################################################################################################
"""
Fit an LSTM network to a 2-D time series prediction
"""
def test_LSTM_model():
    dimension = 3
    train_series = generate_trig_series(dimension, (0, 2*math.pi))
    plot_series(train_series, "Training series")

    # LSTM Architecture
    input_timesteps = 10
    output_timesteps = 1
    input_layer_units = dimension   # Dimensionality of time series data
    hidden_layer_units = 50
    output_layer_units = dimension * output_timesteps  # We want to simultaneously predict all dimensions of a number of future time points


    # Create network
    model = Sequential()
    model.add(LSTM(hidden_layer_units, input_shape=(input_timesteps, input_layer_units)))
    model.add(Dense(output_layer_units))
    model.compile(loss='mae', optimizer='adam')

    # Training params
    batch_size = 20 # Mini batch size in GD/ other algorithm
    epcohs = 20 # 50 is good

    # Train network
    X, Y = gen.prepare_dataset(train_series, input_timesteps, output_timesteps)

    history = model.fit(X, Y, epochs=epcohs, batch_size=batch_size, verbose=2)
                        # , validation_data=(test_X, test_y), shuffle=False)

    # Plot training history
    plt.figure()
    plt.title("Training history")
    plt.plot(history.history['loss'], label='training_loss')
    # plt.plot(history.history['val_loss'], label='test')
    plt.legend()


    # Predict on test data
    test_t_range = (-2*math.pi, 2*math.pi)
    test_series = generate_trig_series(dimension, test_t_range)
    X_test, Y_test = gen.prepare_dataset(test_series, input_timesteps, output_timesteps)

    Y_predicted = model.predict(X_test)

    # Seperate out predicted multiple timeseries (for multiple output timesteps)
    Y_predicted_multi_series = seperate_multi_timestep_series(Y_predicted, dimension, output_timesteps)
    Y_true_multi_series = seperate_multi_timestep_series(Y_test, dimension, output_timesteps)

    # Plot each of above output series
    for i in range(output_timesteps):
        predicted_series = Y_predicted_multi_series[i]
        true_series = Y_true_multi_series[i]

        predicted_dp_series = gen.convert_to_datapoint_series(predicted_series, test_t_range)
        true_dp_series = gen.convert_to_datapoint_series(true_series, test_t_range)

        plt.figure()
        plot_series(true_dp_series, "Y_true")
        plot_series(predicted_dp_series, "Y_predicted")
        plt.title("Test prediction")


    plt.show()



###################################################################################################

# Call test functions

test_LSTM_model()