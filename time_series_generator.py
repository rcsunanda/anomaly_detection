"""
TimeSeriesGenerator class
"""

import anomaly_detection.data_point as data_point
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# Generate time series of given size
def generate_time_series(dim, t_range, count, functions, anomaly_rate=0, add_noise=False, noise_var=1):
    assert (len(t_range) == 2)
    assert (len(functions) == dim)

    t_vals = np.linspace(t_range[0], t_range[1], num=count)
    series = []
    # Sample value for each dimension
    for idx in range(count):
        sample_X = np.zeros((dim,))
        t = t_vals[idx]

        for d in range(dim):
            func = functions[d]
            dim_sample_val = func(t, d)
            sample_X[d] = dim_sample_val

        if add_noise:
            noise_vec = np.random.multivariate_normal(np.zeros(dim), np.eye(dim)*noise_var)
            sample_X = np.add(sample_X, noise_vec)

        series.append(data_point.DataPoint(t, sample_X, False, False))

    # Set anomolous points

    anomaly_count = int(count * anomaly_rate)
    anomalous_point_indexes = np.random.randint(low=0, high=count, size=anomaly_count)
    for an_idx in anomalous_point_indexes:
        point = series[an_idx]

        low = np.add(point.X, np.ones(dim))
        high = np.subtract(point.X, np.ones(dim))
        deviation_vec = np.random.uniform(low, high)

        point.X = np.add(point.X, deviation_vec)
        point.true_is_anomaly = True

    return series


def scale_series(series):

    dataset_size = len(series)
    data_dim = len(series[0].X)

    X = np.zeros((dataset_size, data_dim))

    for i, point in enumerate(series):
        X[i] = point.X

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(X)

    for i, point in enumerate(series):
        point.X = scaled[i]


def prepare_dataset(series_points, input_timesteps, output_timesteps):

    dataset_size = len(series_points)
    data_dim = len(series_points[0].X)

    # Cannot go beyond (not enough datapoints to gather past and future points)
    output_dataset_size = dataset_size - input_timesteps - output_timesteps

    X = np.zeros((output_dataset_size, input_timesteps, data_dim))
    Y = np.zeros((output_dataset_size, data_dim * output_timesteps))

    for idx in range(output_dataset_size):
        # Prepare X with current and previous timesteps
        step = 0
        for j in range(idx, idx + input_timesteps):
            point = series_points[j]
            for dim in range(data_dim):
                X[idx][step][dim] = point.X[dim]
            step += 1

        # Prepare Y with next timesteps
        future_start = idx + input_timesteps
        step = 0
        for k in range(future_start, future_start + output_timesteps):
            future_point = series_points[k]
            for dim in range(data_dim):
                Y[idx][step+dim] = future_point.X[dim]
            step += 1

    # X is numpy array with shape (dataset_size, input_timesteps, data_dim)
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

