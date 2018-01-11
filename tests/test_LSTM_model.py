"""
Tests for LSTMNetwork
"""

import anomaly_detection.time_series_generator as gen
import math
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


"""
Helper function
"""
# range is a tuple with (start, end) for time variable
def generate_trig_series(range):
    dim1_func = math.sin
    dim2_func = math.cos

    generator = gen.TimeSeriesGenerator(2, [dim1_func, dim2_func])
    print(generator)

    series = generator.generate_time_series(range, 1000, 0)
    return generator, series


def plot_series(series, title):
    t = [point.t for point in series]
    x1 = [point.X[0] for point in series]
    x2 = [point.X[1] for point in series]

    plt.plot(t, x1, label=title+'_dimension-1')
    plt.plot(t, x2, label=title+'_dimension-2')

    plt.title(title)
    plt.xlabel("t")
    plt.ylabel("y")
    plt.legend(loc='upper right')


###################################################################################################
"""
Fit an LSTM network to a 2-D time series prediction
"""
def test_LSTM_model():
    generator, train_series = generate_trig_series((0, 2*math.pi))
    plot_series(train_series, "Training series")

    # LSTM Architecture
    input_layer_units = 2   # Dimensionality of time series data
    hidden_layer_units = 50
    output_layer_units = input_layer_units  # We want to simultaneously predict all dimensions of time-series data

    time_steps = 1

    # Create network
    model = Sequential()
    model.add(LSTM(hidden_layer_units, input_shape=(time_steps, input_layer_units)))
    model.add(Dense(output_layer_units))
    model.compile(loss='mae', optimizer='adam')

    # Training params
    batch_size = 20 # Mini batch size in GD/ other algorithm
    epcohs = 20 # 50 is good

    # Train network
    X, Y = generator.prepare_dataset(train_series, time_steps)
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
    gen2, test_series = generate_trig_series(test_t_range)
    X_test, Y_test = gen2.prepare_dataset(test_series, time_steps)

    Y_predicted = model.predict(X_test)

    Y_predicted_series = gen2.convert_to_series(Y_predicted, test_t_range)
    Y_test_series = gen2.convert_to_series(Y_test, test_t_range)
    # plot_series("Y_test")
    plt.figure()
    plot_series(Y_test_series, "Y_true")
    plot_series(Y_predicted_series, "Y_predicted")
    plt.title("Test prediction")


    plt.show()



###################################################################################################

# Call test functions

test_LSTM_model()