"""
Default covariance for probability based optimisation methods.
Designed to be used with default interpretation of parameter (i.e log transform)

Note:
    0.4 factor for covariance can explained in the following way:

        Probability of X > k * m = Probability of (X - m) / m > (k-1) 
        If X is normally distributed N(m, m * sigma), then this is
        Proba N(0,1) > (k-1)/ sigma
        If X is log normal distributed, N(log(m), a), then this it
        Proba N(0,1) > log(k) /a
        Hence for the probabilities to match, a = sigma log(k)/(k-1)

        Take k = 3 to conclude. 
"""

import numpy as np

# Default covariance in log space
# (.4 is to keep probability of outputing large values under control)
default_cov = (0.4 * np.diag([0.5, 1.2, 0.8, 1.2, 1.0])) ** 2
