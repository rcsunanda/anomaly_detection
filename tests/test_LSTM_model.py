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
from keras.utils.vis_utils import plot_model
from sklearn.metrics import mean_squared_error

"""
Helper function
"""
# range is a tuple with (start, end) for time variable
def generate_trig_series(dim, t_range):
    def dim1_func(x, d):
        return math.sin(x)

    def dim2_func(x, d):
        return math.cos(x)

    def dim3_func(x, d):
        theta = math.pi / 6
        return 3*math.sin(x+theta)

    if dim == 1:
        functions = [dim1_func]
    elif dim == 2:
        functions = [dim1_func, dim2_func]
    elif dim == 3:
        functions = [dim1_func, dim2_func, dim3_func]
    else:
        assert False

    series = gen.generate_time_series(dim=dim, t_range=t_range, count=1000,
                                      functions=functions, is_anomolous=0,
                                      add_noise=True, noise_var=0.01)

    return series


def generate_complex_series(dim, t_range):

    def damped_sine(x, d):
        return (math.e**x) * (math.sin(2 * math.pi * x))

    def cubic_func(x, d):
        return x**3 - 6 * x**2 + 4*x + 12

    def freq_increasing_sine(x, d):
        return math.sin(2 * math.pi * math.e**x)

    if dim == 1:
        functions = [damped_sine]
    elif dim == 2:
        functions = [damped_sine, cubic_func]
    elif dim == 3:
        functions = [damped_sine, cubic_func, freq_increasing_sine]
    else:
        assert False

    series = gen.generate_time_series(dim=dim, t_range=t_range, count=1000,
                                      functions=functions, is_anomolous=0,
                                      add_noise=True, noise_var=0.01)

    return series


def generate_high_dim_complex_series(dim, t_range):

    def damped_sine(x, d):
        return (d+1) * (math.e**x) * (math.sin(2 * math.pi * x))

    def cubic_func(x, d):
        return (d+1) * x**3 - (d+1) * 6 * x**2 + 4*x + 12

    def freq_increasing_sine(x, d):
        return (d+1) * math.sin(2 * math.pi * math.e**x)

    all_functions = [damped_sine, cubic_func, freq_increasing_sine]

    functions = []
    for d in range(dim):
        func_index = d % 3
        functions.append(all_functions[func_index])

    series = gen.generate_time_series(dim=dim, t_range=t_range, count=1000,
                                      functions=functions, is_anomolous=0,
                                      add_noise=True, noise_var=0.01)

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


###################################################################################################
"""
Fit an LSTM network to a 2-D time series prediction
"""
def test_LSTM_model():

    time_series_gen_function = generate_high_dim_complex_series
    trig_train_t_range = (-math.pi, math.pi)
    complex_train_t_range = (-1, 1)

    dimension = 50
    train_series = time_series_gen_function(dimension, complex_train_t_range)
    plot_series(train_series, "Training series")

    # LSTM Architecture
    input_timesteps = 4
    output_timesteps = 1
    input_layer_units = dimension   # Dimensionality of time series data
    hidden_layer_units = 50
    output_layer_units = dimension * output_timesteps  # We want to simultaneously predict all dimensions of a number of future time points


    # Create network
    model = Sequential()
    model.add(LSTM(hidden_layer_units, input_shape=(input_timesteps, input_layer_units)))
    # model.add(LSTM(hidden_layer_units, return_sequences=True))
    model.add(Dense(output_layer_units))
    model.compile(loss='mae', optimizer='adam')

    print(model.summary())
    # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    # Training params
    batch_size = 20 # Mini batch size in GD/ other algorithm
    epcohs = 20 # 50 is good

    # Train network
    gen.scale_series(train_series)
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

    trig_test_t_range = (-2*math.pi, 2*math.pi)
    complex_test_t_range = (-2, 2)
    test_t_range = complex_test_t_range

    test_series = time_series_gen_function(dimension, test_t_range)
    gen.scale_series(test_series)
    X_test, Y_test = gen.prepare_dataset(test_series, input_timesteps, output_timesteps)

    Y_predicted = model.predict(X_test)

    # Seperate out predicted multiple timeseries (for multiple output timesteps)
    Y_predicted_multi_series = gen.seperate_multi_timestep_series(Y_predicted, dimension, output_timesteps)
    Y_true_multi_series = gen.seperate_multi_timestep_series(Y_test, dimension, output_timesteps)

    # Plot each of above output series
    for i in range(output_timesteps):
        predicted_series = Y_predicted_multi_series[i]
        true_series = Y_true_multi_series[i]

        rmse = math.sqrt(mean_squared_error(true_series, predicted_series))
        print("output_series_no={}, rmse={}".format(i, rmse))

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

