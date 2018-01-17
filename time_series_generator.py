"""
TimeSeriesGenerator class
"""

import anomaly_detection.data_point as data_point
import numpy as np


# Generate time series of given size
def generate_time_series(dim, t_range, count, functions, is_anomolous):
    assert (len(t_range) == 2)
    assert (len(functions) == dim)

    t_vals = np.linspace(t_range[0], t_range[1], num=count)
    series = []
    # Sample value for each dimension
    for idx in range(count):
        sample_X = []
        t = t_vals[idx]

        for d in range(dim):
            func = functions[d]
            dim_sample_val = func(t)
            sample_X.append(dim_sample_val)

        series.append(data_point.DataPoint(t, sample_X, is_anomolous, -1))

    return series


def prepare_dataset(series_points, time_steps):

    dataset_size = len(series_points)
    data_dim = len(series_points[0].X)

    X = np.zeros((dataset_size, data_dim))
    Y = np.zeros((dataset_size, data_dim))

    # Append a copy last element to series_points (to prepare Y)
    last_point = series_points[-1]
    series_points.append(data_point.DataPoint(
        last_point.t, last_point.X, last_point.true_is_anomaly, last_point.predicted_is_anomaly))

    for idx in range(dataset_size):
        curr_point = series_points[idx]
        next_point = series_points[idx + 1]
        for dim in range(data_dim):
            X[idx][dim] = curr_point.X[dim]
            Y[idx][dim] = next_point.X[dim]

    X = X.reshape((dataset_size, time_steps, data_dim))
    Y = Y.reshape((dataset_size, data_dim))  ## Is this necessary? Isn't Y already in this shape?

    # X is numpy array with shape (dataset_size, time_steps, data_dim)
    # Y is numpy array with shape (dataset_size, data_dim)
    return (X, Y)


# series is a n-dimensional time series in a numpy ndarray format
# Return our standard time series type (list of DataPoints)
def convert_to_datapoint_series(input_series, t_range):
    assert (len(t_range) == 2)
    dataset_size = input_series.shape[0]
    data_dim = input_series.shape[1]

    t_vals = np.linspace(t_range[0], t_range[1], num=dataset_size)

    output_series = []

    for idx in range(dataset_size):
        sample_X = []
        t = t_vals[idx]

        for dim in range(data_dim):
            dim_sample_val = input_series[idx][dim]
            sample_X.append(dim_sample_val)

        output_series.append(data_point.DataPoint(t, sample_X, -1, -1))

    return output_series
