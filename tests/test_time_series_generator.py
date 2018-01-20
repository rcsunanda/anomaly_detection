"""
Tests for TimeSeriesGenerator
"""

import anomaly_detection.data_synthesizer as ds
import anomaly_detection.data_formulation as df
import math
import matplotlib.pyplot as plt


###################################################################################################
"""
Generate a 2-D time series and plot
"""

def test_generate_time_series():

    dim = 3
    series = ds.generate_high_dim_complex_series(dim=dim, t_range=(-1, 1), count=1000, anomaly_rate=0.05)

    df.plot_series(series, 'Test_series', dim)

    plt.show()


###################################################################################################
"""
Visualize dataset in 3D
"""

def test_dataset_visualization():

    dim = 3
    series = ds.generate_high_dim_complex_series(dim=dim, t_range=(-1, 1), count=1000, anomaly_rate=0.05)

    df.plot_series(series, 'Test_series', dim)

    df.visualize_dataset(series)

    plt.show()


###################################################################################################

# Call test functions

# test_generate_time_series()
test_dataset_visualization()