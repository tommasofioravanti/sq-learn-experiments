import numpy as np
from sklearn.QuantumUtility.Utility import *

epsilon = 0.1
a_list = [0, 1]

# Set M to even value to estimate theta=1 with certainity.
estimations = [amplitude_estimation(a, M=158, epsilon=epsilon) for a in a_list]

print(estimations)
