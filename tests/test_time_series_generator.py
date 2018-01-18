"""
Tests for TimeSeriesGenerator
"""

import anomaly_detection.time_series_generator as gen
import math
import matplotlib.pyplot as plt


###################################################################################################
"""
Generate a 2-D time series and plot
"""

def test_generate_time_series():

    dim1_func = math.sin
    dim2_func = math.cos

    series = gen.generate_time_series(dim=2, t_range=(0, 2*math.pi), count=100,
                                      functions=[dim1_func, dim2_func], is_anomolous=0,
                                      add_noise = True, noise_var = 0.01)

    # Plot
    t = [point.t for point in series]
    x1 = [point.X[0] for point in series]
    x2 = [point.X[1] for point in series]
    plt.plot(t, x1)
    plt.plot(t, x2)

    plt.show()



###################################################################################################

# Call test functions

test_generate_time_series()