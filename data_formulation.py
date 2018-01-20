"""
TimeSeriesGenerator class
"""

import anomaly_detection.data_point as data_point
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition

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
# original_series, if given, is used to take t and anomaly values
# Return our standard time series type (list of DataPoints)
def convert_to_datapoint_series(input_series, t_range, original_series=None):
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

        true_is_anomaly, predicted_is_anomaly = False, False
        if original_series:
            orig_point = original_series[idx]
            t = orig_point.t
            true_is_anomaly = orig_point.true_is_anomaly
            predicted_is_anomaly = orig_point.predicted_is_anomaly

        output_series.append(data_point.DataPoint(t, sample_X, true_is_anomaly, predicted_is_anomaly))

    return output_series


def covert_to_standard_dataset(series):
    dataset_size = len(series)
    dim = len(series[0].X)
    X = np.zeros((dataset_size, dim))

    for idx, point in enumerate(series):
        for d in range(dim):
            X[idx][d] = point.X[d]

    return X

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


def evaluate_detection_results(id, series):
    total_accurate, true_positives, false_negatives, false_positives, true_negatives = (0,) * 5
    actual_positives, actual_negatives = (0,) * 2
    for point in series:
        actual, predicted = point.true_is_anomaly, point.predicted_is_anomaly

        if actual is True and predicted is True:
            true_positives += 1

        if actual is True and predicted is False:
            false_negatives += 1

        if actual is False and predicted is True:
            false_positives += 1

        if actual is False and predicted is False:
            true_negatives += 1

        if actual == predicted:
            total_accurate += 1

        if actual is True:
            actual_positives += 1

    total = len(series)
    actual_negatives = total - actual_positives

    overall_accuracy_percentage = total_accurate * 100 / total
    true_positive_percentage = true_positives * 100 / actual_positives
    false_negative_percentage = false_negatives * 100 / actual_negatives
    true_negative_percentage = true_negatives * 100 / actual_negatives
    false_positives_percentage = false_positives * 100 / actual_negatives
    precision = true_positives * 100 / true_positives + false_positives
    F1_score = (2 * true_positives) / (2 * true_positives + false_positives + false_negatives)

    print("series_id={}"
          "\n\t overall_accuracy_percentage={:.2f}%"
          "\n\t precision={:.2f}%"
          "\n\t true_positives_percentage (sensitivity) (correctly detected anomalies)={:.2f}%"
          "\n\t false_negative_percentage (undetected anomalies)={:.2f}%"
          "\n\t false_positives_percentage (incorrectly detected anomalies: false warning)={:.2f}%"
          "\n\t true_negative_percentage (specificity) (correctly detected normal data points)={:.2f}%"
          "\n\t F1_score={:.2f}"
          .format(id, overall_accuracy_percentage, precision, true_positive_percentage,
                  false_negative_percentage, false_positives_percentage, true_negative_percentage, F1_score))


def plot_series(series, title, max_dim_to_plot):

    dim = len(series[0].X)
    t = [point.t for point in series]

    for d in range(dim):
        if d > max_dim_to_plot:
            break;

        x = [point.X[d] for point in series]
        # plt.plot(t, x, label=title + '_dimension-' + str(d))
        plt.plot(x, label=title + '_dimension-' + str(d))

    plt.title(title)
    plt.xlabel("t")
    plt.ylabel("y")
    plt.legend(loc='upper right')


def visualize_dataset(series):
    assert len(series[0].X) == 3    # There must only be 3 dimensions for visualization

    normal_xs, normal_ys, normal_zs = [], [], []
    anormalous_xs, anormalous_ys, anormalous_zs = [], [], []

    for point in series:
        xs, ys, zs = point.X[0], point.X[1], point.X[2]
        if point.true_is_anomaly:
            anormalous_xs.append(xs)
            anormalous_ys.append(ys)
            anormalous_zs.append(zs)
        else:
            normal_xs.append(xs)
            normal_ys.append(ys)
            normal_zs.append(zs)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(normal_xs, normal_ys, normal_zs, c='b', marker='o')
    ax.scatter(anormalous_xs, anormalous_ys, anormalous_zs, c='r', marker='^')
    plt.figure()


def do_PCA(X, num_components):
    pca = decomposition.PCA(n_components=num_components)
    pca.fit(X)
    X_reduced = pca.transform(X)
    return X_reduced