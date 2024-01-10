#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 09:28:56 2024

@author: bettina

Next steps: 
    - D1_Bolhuis: introduce parameters for position of maximum and minima
    - D1_Bolhuis: plot potential for various parameters
    - D1_Bolhuis: check whether paramters can be passed as a list or array, e.g. D1_Bolhuis.V(x, my_param)
    
    - D1_Bohuis: implement force, analytically
    - D1_Bohuis: implement force, numerically via finite difference
    
    - D1_Bolhuis: implement unnormalized Boltzmann dist
    - D1_Bolhuis: implement partition function
"""

#-----------------------------------------
#   I M P O R T S 
#-----------------------------------------
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import scipy.constants as const

# local packages and modules
from potentials import D1_Bolhuis

print(D1_Bolhuis.V(1,1,1,1))