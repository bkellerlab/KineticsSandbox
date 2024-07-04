"""
Created on Wed July 3 13:45:56 2024

@author: arthur

This tests the one-dimensional Morese and Bolhuis potential 
implemented in the potential class

"""

import numpy as np
from general_benchmark_potentials_D1 import benchmark_1d_potential

from potential.D1 import Bolhuis, Morse

param_ranges = {
    "a": np.linspace(-2, 2, 11),
    "b": np.linspace(0, 4, 11),
    "c": np.linspace(0, 40, 11),
    "k1": np.linspace(0, 4, 11),
    "k2": np.linspace(0, 4, 11),
    "alpha": np.linspace(0, 4, 11),
}

param_names = ["a", "b", "c", "k1", "k2", "alpha"]
x_range = (-3.5, 3.5, 500)
h = 0.01

benchmark_1d_potential(Bolhuis, param_ranges, param_names, x_range, h)

param_ranges = {
    "D_e": np.linspace(1, 10, 5),
    "a": np.linspace(0.5, 2.0, 5),
    "x_e": np.linspace(0.75, 1.25, 5),
}
param_names = ["D_e", "a", "x_e"]
x_range = (0.5, 2.5, 100)
h = 0.01

benchmark_1d_potential(Morse, param_ranges, param_names, x_range, h)
