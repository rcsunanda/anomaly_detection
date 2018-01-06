"""
Process class
"""

import anomaly_detection.data_point as data_point

import scipy.stats as st
import numpy as np
import math


###################################################################################################
"""
Generates a time series from given multivariable function
"""

class TimeSeriesGenerator:
    def __init__(self, num_dimensions, functions):
        self.num_dimensions = num_dimensions
        self.set_functions(functions)


    def __repr__(self):
        return "TimeSeriesGenerator(\n\tnum_dimensions={} \n\tfunctions={} \n)"\
            .format(self.num_dimensions, self.functions)


    def set_functions(self, functions):
        assert (len(functions) == self.num_dimensions)
        self.functions = functions  # Function for each dimension used for generating time series


    # Generate time series of given size
    def generate_time_series(self, t_range, count, is_anomolous):
        assert (len(t_range) == 2)
        t_vals = np.linspace(t_range[0], t_range[1], num=count)
        series = []
        # Sample value for each dimension
        for idx in range(count):
            sample_X = []
            t = t_vals[idx]

            for dim in range(self.num_dimensions):
                func = self.functions[dim]
                dim_sample_val = func(t)
                sample_X.append(dim_sample_val)

            series.append(data_point.DataPoint(t, sample_X, is_anomolous, -1))

        return series



