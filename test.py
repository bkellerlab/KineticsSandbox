#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 09:28:56 2024

@author: bettina

Next steps:
     
    #------------------------------------------
    # package "potentials" 

    - D1: DoubleWell
    - D1: GaussianBias
    
    #------------------------------------------
    # package "rate_theory" 
    
    np.- create a new package "rate_theory" 
    - within rate_theory, write a module D1 for rates from 1D-potentials
    - D1: TST
    - D1: Kramers
    - D1: SqRA
    - within rate_theory, write a module rate_matrix for rates from rate_matrices
    - rateMatrix: SqRA_rate (via Berezhkovski, Szabo)    
    - rateMatrix: SqRA_rate_via_its (as inverse of ITS)    
    - rateMatrix: MSM_rate_via_its (as inverse of ITS)   

    #------------------------------------------
    # create a new package "integrators" 


"""

#-----------------------------------------
#   I M P O R T S 
#-----------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from scipy.optimize import minimize



