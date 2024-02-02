#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 05:14:50 2024

@author: bettina
"""

#-----------------------------------------
#   I M P O R T S 
#-----------------------------------------
import sys
sys.path.append("..")  

import numpy as np
import scipy.constants as const

# local packages and modules
from potential import D1
#from system import system



def TST_D1(x_A, x_TS, T, m, potential): 
    """
    Calculate the rate constant using Transition State Theory (TST) for a unimolecular reaction.

    Parameters:
    - x_A (float): Position of reactant A.
    - x_TS (float): Position of the transition state (TS).
    - T (float): Temperature in Kelvin.
    - m (float): Mass of the reacting molecule.
    - potential (object): Object representing the potential energy surface, with methods:
        - potential.potential(x): Calculate the potential energy at position x.
        - potential.hessian(x): Calculate the Hessian matrix at position x.

    Returns:
    - k_AB (float): Rate constant for the reaction from A to B.

    The TST formula used is based on the assumption of a harmonic transition state and the 
    separation of reactant and transition state motions. The vibrational partition function 
    of the transition state (q_TS) is set to one, and the partition function of A (q_A) is 
    calculated based on the vibrational frequency and temperature. The energy barrier is 
    calculated as (E_b) as the difference in the potential at the transition state and at
    the reactant state.

    The rate constant (k_AB) is then computed using the Eyring TST
    
    k_AB = R*T / h *  (q_TS / q_A) * np.exp(- E_b / (R * T)) 

    The unit of k_AB is 1/ps

    Note: This function assumes a one-dimensional reaction coordinate.
    """
    
    # get natural constants in the right units    
    h = const.h * const.Avogadro * 0.001 *  1e12
    R = const.R * 0.001
    
    # extract positions of A and TS, temperature and mass
    
    # get the energy barrier
    E_b = potential.potential(x_TS) - potential.potential(x_A)

    #get the force consant in the transition state
    k = potential.hessian(x_A)[0,0]
    
    # calculate the frequency
    nu = 1/(2* np.pi) * np.sqrt(k/m)
    
    #set vibrational partition function of the TS to one
    q_TS = 1
    
    # calculate the partition function of A
    q_A = np.exp(- h * nu / (2 * R * T)) /  (1 - np.exp(- h * nu / ( R * T)) )
    
    k_AB = R * T / h * (q_TS / q_A) * np.exp(- E_b / (R * T))    
    
    return k_AB


