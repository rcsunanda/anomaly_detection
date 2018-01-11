"""
LSTMNetwork class
"""

import anomaly_detection.data_point as data_point

import scipy.stats as st
import numpy as np
import math


###################################################################################################
"""
An LSTM network model
"""

class LSTMNetwork:
    def __init__(self, num_dimensions, functions):
        self.num_dimensions = num_dimensions
        self.set_functions(functions)


    def __repr__(self):
        return "TimeSeriesGenerator(\n\tnum_dimensions={} \n\tfunctions={} \n)"\
            .format(self.num_dimensions, self.functions)


