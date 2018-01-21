"""
Tests for LSTMNetwork
"""

import anomaly_detection.data_formulation as df
import anomaly_detection.utility as utility
import anomaly_detection.sys_params as sp
import anomaly_detection.data_point as dp

import math
import matplotlib.pyplot as plt
import numpy as np
import csv

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils.vis_utils import plot_model
from sklearn.metrics import mean_squared_error


def get_synthetic_training_data(sys_params):
    print("get_synthetic_training_data: Generating synthetic training data")

    train_series = sys_params.data_generation_func(sys_params.dimension,
                                                   sys_params.train_data_t_range, 1000, anomaly_rate=0)
    return train_series


def get_synthetic_test_data(sys_params):
    print("get_synthetic_test_data: Generating synthetic test data, and prepare for predicting")

    input_timesteps = sys_params.input_timesteps
    output_timesteps = sys_params.output_timesteps

    test_series = sys_params.data_generation_func(sys_params.dimension,
                                                  sys_params.test_data_t_range, 1000, sys_params.anomaly_rate)
    df.scale_series(test_series)
    X_test, Y_test = df.prepare_dataset(test_series, input_timesteps, output_timesteps)

    # From the original test_series, some points are removed in prepare_dataset
    #   (input_timesteps points from the beginning and output_timesteps points from the end)
    # To correctly align anomaly points with Y_test and Y_predicted, we must remove those points from original test_series
    truncated_test_series = test_series[input_timesteps: -output_timesteps]

    return X_test, Y_test, truncated_test_series


def load_series_from_file(filename):
    with open(filename, newline='') as file:
        reader = csv.reader(file)
        series = []
        for i, row in enumerate(reader):
            if i > 0:   # Ignore first row with column names
                sample_X = np.array(row[1:])    # Ignore timestamp
                series.append(dp.DataPoint(i, sample_X, False, False))

    return series


def get_real_training_data(sys_params):
    print("get_real_training_data: Loading real training data and sanitizing. filename={}".format(sys_params.training_data_file))
    series = load_series_from_file(sys_params.training_data_file)
    sys_params.dimension = len(series[0].X)  # Ignore timestamp column
    print("\t sys_params.dimension={}".format(sys_params.dimension))
    df.sanitize_series(series)
    return series


def get_real_test_data(sys_params):
    print("get_real_test_data: Loading real test data and preparing for predicting. filename={}".
          format(sys_params.test_data_file))

    input_timesteps = sys_params.input_timesteps
    output_timesteps = sys_params.output_timesteps

    test_series = load_series_from_file(sys_params.test_data_file)

    df.sanitize_series(test_series)

    sys_params.test_data_t_range = (1, len(test_series))

    test_series[int(len(test_series)/2)].true_is_anomaly = True   # Temporary; so that actual_positives > 0 (avoid division by zero)

    df.scale_series(test_series)
    X_test, Y_test = df.prepare_dataset(test_series, input_timesteps, output_timesteps)
    truncated_test_series = test_series[input_timesteps: -output_timesteps]

    return X_test, Y_test, truncated_test_series


def get_training_data(sys_params):
    if sys_params.system_name == 'synthetic_dataset':
        return get_synthetic_training_data(sys_params)
    elif sys_params.system_name == 'real_dataset_1':
        return get_real_training_data(sys_params)


def get_test_data(sys_params):
    if sys_params.system_name == 'synthetic_dataset':
        return get_synthetic_test_data(sys_params)
    elif sys_params.system_name == 'real_dataset_1':
        return get_real_test_data(sys_params)


def build_model(sys_params):

    print("build_model: Initializing and compiling LSTM model")

    # LSTM network architecture
    input_timesteps = sys_params.input_timesteps
    output_timesteps = sys_params.output_timesteps
    input_layer_units = sys_params.dimension   # Dimensionality of time series data
    hidden_layer_units = sys_params.hidden_layer_units
    output_layer_units = sys_params.dimension * output_timesteps  # We want to simultaneously predict all dimensions of a number of future time points

    model = Sequential()

    # Add layers
    model.add(LSTM(hidden_layer_units, input_shape=(input_timesteps, input_layer_units)))
    # model.add(LSTM(hidden_layer_units, return_sequences=True))
    model.add(Dense(output_layer_units))

    model.compile(loss='mae', optimizer='adam')

    return model


def train_model(sys_params, model, train_series):

    print("train_model: Starting LSTM model training")

    df.scale_series(train_series)
    df.plot_series(train_series, "Training series", sys_params.max_dim_to_plot)

    X, Y = df.prepare_dataset(train_series, sys_params.input_timesteps, sys_params.output_timesteps)

    history = model.fit(X, Y, epochs=sys_params.epcohs, batch_size=sys_params.batch_size, verbose=2)
                        # , validation_data=(test_X, test_y), shuffle=False)

    # Plot training history
    plt.figure()
    plt.title("Training history")
    plt.plot(history.history['loss'], label='training_loss')
    # plt.plot(history.history['val_loss'], label='test')
    plt.legend()

    print("train_model: Finished LSTM model training")


def detect_anomalies(sys_params, Y_predicted, Y_test, truncated_test_series):

    print("detect_anomalies: Detecting anomalies")

    # Seperate out predicted multiple timeseries (for multiple output timesteps)

    Y_predicted_multi_series = df.seperate_multi_timestep_series(Y_predicted, sys_params.dimension, sys_params.output_timesteps)
    # Y_true_multi_series = df.seperate_multi_timestep_series(Y_test, dimension, output_timesteps)

    # Detect anomalies considering each time series one by one -> ToDo: Improve (eg: errror vector method)
    for i in range(sys_params.output_timesteps):
        predicted_series = Y_predicted_multi_series[i]

        diff = np.subtract(predicted_series, Y_test)
        distance = np.linalg.norm(diff, axis=1) # Distance of diff vector at each time point

        rmse = math.sqrt(mean_squared_error(Y_test, predicted_series))
        print("output_series_no={}, rmse={}".format(i, rmse))

        for j, point in enumerate(truncated_test_series):
            if distance[j] > sys_params.diff_anomaly_threshold:
                point.predicted_is_anomaly = True

        df.evaluate_detection_results(i, truncated_test_series)


def draw_plots(sys_params, Y_predicted, Y_test, truncated_test_series):
    print("draw_plots: Drawing plots (true vs predicted series and anomalies)")

    Y_predicted_multi_series = df.seperate_multi_timestep_series(Y_predicted, sys_params.dimension, sys_params.output_timesteps)

    # Plot each of output series corresponding to each output_timesteps
    for i in range(sys_params.output_timesteps):
        predicted_series = Y_predicted_multi_series[i]

        diff = np.subtract(predicted_series, Y_test)
        distance = np.linalg.norm(diff, axis=1)  # Distance of diff vector at each time point

        predicted_dp_series = df.convert_to_datapoint_series(predicted_series, sys_params.test_data_t_range)

        # True, vs predicted multi-dimensional time-series plot

        plt.figure()
        df.plot_series(truncated_test_series, "Y_true", sys_params.max_dim_to_plot)
        df.plot_series(predicted_dp_series, "Y_predicted", sys_params.max_dim_to_plot)
        plt.title("Test prediction")

        for j, point in enumerate(truncated_test_series):
            if point.true_is_anomaly:
                plt.axvline(j, color='#550000ff', linewidth=0.5, label='Anomaly')

        # Distance plot

        plt.figure()
        plt.plot(distance)
        plt.title("Distance(true, predicted)")

        plt.axhline(sys_params.diff_anomaly_threshold, label='threshold', color='k', linestyle='--')

        for j, point in enumerate(truncated_test_series):
            if point.true_is_anomaly:
                plt.axvline(j, color='#550000ff', linewidth=0.5, label='Anomaly')

    plt.show()


def do_pca_and_visualize(sys_params, series):
    print("do_pca_and_visualize: Running PCA for 3 components, and visualizing in 3D space")

    dim = len(series[0].X)
    if (dim > 3):
        X = df.covert_to_standard_dataset(series)
        X_reduced = df.do_PCA(X, num_components=3)
        t_range = (1, len(X_reduced))  # Unused
        X_reduced_series = df.convert_to_datapoint_series(X_reduced, t_range, series)
    else:
        X_reduced_series = series

    df.visualize_dataset(X_reduced_series)


def run_system(system_name):

    sys_params = sp.init_system_params(system_name)

    training_series = get_training_data(sys_params)

    do_pca_and_visualize(sys_params, training_series)

    model = build_model(sys_params)

    train_model(sys_params, model, training_series)

    X_test, Y_test, truncated_test_series = get_test_data(sys_params)

    Y_predicted = model.predict(X_test)

    detect_anomalies(sys_params, Y_predicted, Y_test, truncated_test_series)

    draw_plots(sys_params, Y_predicted, Y_test, truncated_test_series)


if __name__ == "__main__":
    # run_system("synthetic_dataset")
    run_system("real_dataset_1")
