from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import anomaly_detection.data_formulation as gen
import math
import numpy as np
import anomaly_detection.data_point as data_point


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def prepare_dataset(series_points):

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

    # X = X.reshape((dataset_size, time_steps, data_dim))
    # Y = Y.reshape((dataset_size, data_dim))  ## Is this necessary? Isn't Y already in this shape?

    # X is numpy array with shape (dataset_size, time_steps, data_dim)
    # Y is numpy array with shape (dataset_size, data_dim)
    return np.concatenate((X, Y), axis=1)


# range is a tuple with (start, end) for time variable
def generate_trig_series(dim, t_range):
    counter = 0
    def dim1_func(x):
        nonlocal counter
        counter += 1
        return counter

    counter_100 = 100
    def dim2_func(x):
        nonlocal counter_100
        counter_100 += 1
        return counter_100

    counter_1000 = 1000
    def dim3_func(x):
        nonlocal counter_1000
        counter_1000 += 1
        return counter_1000

    if (dim == 1):
        functions = [dim1_func]
    elif (dim == 2):
        functions = [dim1_func, dim2_func]
    elif (dim == 3):
        functions = [dim1_func, dim2_func, dim3_func]
    else:
        assert(False)

    dataset_size = 12
    series = gen.generate_time_series(dim=dim, t_range=t_range,
                                      count=dataset_size, functions=functions, is_anomolous=0)

    X = np.zeros((dataset_size, dim))

    for idx in range(dataset_size):
        curr_point = series[idx]
        for d in range(dim):
            X[idx][d] = curr_point.X[d]

    return X


dimension = 2
dataset = generate_trig_series(dim=dimension, t_range=(0, 2*math.pi))
print(dataset)

# dataset = prepare_dataset(train_series)
input_timesteps = 3
output_timesteps = 1
reframed = series_to_supervised(dataset, n_in=input_timesteps, n_out=output_timesteps)

print(reframed)
print("reframed.shape={}".format(np.shape(reframed)))

prepared_dataset = reframed.values

n_obs = input_timesteps * dimension
train_X = prepared_dataset[:, :n_obs]
train_y = prepared_dataset[:, -dimension]

print("before: train_X.shape={}, train_y.shape={}, train_X.len={}".format(train_X.shape, train_y.shape, len(train_X)))

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], input_timesteps, dimension))

print("after: train_X.shape={}, train_y.shape={}".format(train_X.shape, train_y.shape))


pass