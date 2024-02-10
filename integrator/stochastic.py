#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 20:48:01 2024

@author: bettina
"""
#-----------------------------------------
#   I M P O R T S 
#-----------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const

# local packages and modules
#from system import system
#from potential import D1
#from utils import rate_theory


# A step
def A_step(system, half_step ='False'):
    """
    Perform the A-step in a Langevin splitting integrator

    Parameters:
        - system (object): An object representing the physical system undergoing Langevin integration.
                  It should have attributes 'x' (position), 'v' (velocity), and 'dt' (time step).
        - half_step (bool, optional): If True, perform a half-step integration. Default is False, performing
                              a full-step integration. A half-step is often used in the velocity
                              Verlet algorithm for symplectic integration.

    Returns:
        None: The function modifies the 'x' (position) attribute of the provided system object in place.
    """
 
    # set time step, depending on whether a half- or full step is performed
    if half_step == True:
        dt = 0.5 * system.dt
    else:
        dt = system.dt
        
    system.x = system.x + system.v * dt
    
    return None

# B step
def B_step(system, potential, half_step ='False'):
    """
    Perform a Langevin integration B-step for a given system.

    Parameters:
    - system (object): An object representing the physical system undergoing Langevin integration.
                      It should have attributes 'v' (velocity), 'm' (mass), and 'x' (position).
    - potential (object): An object representing the potential energy landscape of the system.
                         It should have a 'force' method that calculates the force at a given position.
    - half_step (bool, optional): If True, perform a half-step integration. Default is False, performing
                                  a full-step integration. A half-step is often used in the velocity
                                  Verlet algorithm for symplectic integration.

    Returns:
    None: The function modifies the 'v' (velocity) attribute of the provided system object in place based on the
          force calculated by the provided potential object.
    """
    
    # set time step, depending on whether a half- or full step is performed
    if half_step == True:
        dt = 0.5 * system.dt
    else:
        dt = system.dt
    
    system.v = system.v + (1 / system.m) * dt * potential.force_num( system.x, 0.001 )[0]
    
    return None

# O_step
def O_step(system, half_step ='False', eta_k = 'None'):
    
    """
    Perform the O-step in a Langevin integrator.

     Parameters:
         - system (object): An object representing the physical system undergoing Langevin integration.
                   It should have attributes 'v' (velocity), 'm' (mass), 'xi' (friction coefficient),
                   'T' (temperature), 'dt' (time step).
         - half_step (bool, optional): If True, perform a half-step integration. Default is False, performing
                   a full-step integration. 
         - eta_k (float or None, optional): If provided, use the given value as the random noise term (eta_k)
                   in the Langevin integrator. If None, draw a new value from a Gaussian normal distribution.

     Returns:
     None: The function modifies the 'v' (velocity) attribute of the provided system object in place.
     """

    # get natural constants in the appropriate units    
    R = const.R * 0.001
    
    # set time step, depending on whether a half- or full step is performed
    if half_step == True:
        dt = 0.5 * system.dt
    else:
        dt = system.dt

    # if eta_k is not provided, draw eta_k from Gaussian normal distribution
    if eta_k == 'None':
        eta_k = np.random.normal()

    d = np.exp(- system.xi * dt)
    f_v = np.sqrt( R * system.T *  (1 / system.m)  * (1 - np.exp(-2 * system.xi * dt)) )

    system.v = d * system.v +  f_v * eta_k
    return None

# ABO integrator
def ABO(system, potential, eta_k = 'None'):
    """
    Perform a full Langevin integration step consisting of A-step, B-step, and O-step.

    Parameters:
    - system (object): An object representing the physical system undergoing Langevin integration.
                      It should have attributes 'x' (position), 'v' (velocity), 'm' (mass), 'xi'
                      (friction coefficient), 'T' (temperature), and 'dt' (time step).
    - potential (object): An object representing the potential energy landscape of the system.
                         It should have a 'force' method that calculates the force at a given position.
    - eta_k (float or None, optional): If provided, use the given value as the random noise term (eta_k)
                        in the Langevin integrator. If None, draw a new value from a Gaussian normal distribution.

    Returns:
    None: The function modifies the 'x' and 'v' attributes of the provided system object in place.
    """    
    
    A_step(system)
    B_step(system, potential)
    O_step(system, eta_k)     
    return None   
    