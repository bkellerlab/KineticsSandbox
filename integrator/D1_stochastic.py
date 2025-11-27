#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 20:48:01 2024

@author: bettina
"""
#-----------------------------------------
#   I M P O R T S 
#-----------------------------------------
import numpy as np
import scipy.constants as const

#---------------------------------------------------------------------
#   O V E R D A M P E D   L A N G E V I N   D Y N A M I C S
#---------------------------------------------------------------------


# Euler-Maruyama integrator
def EM(system, potential, bias=None, eta_k = None, girsanov_reweighting=False):
    """
    Perform a step according to the Euler-Maruyama integrator for overdamped Langevin dynamics

    Parameters:
    - system (object): An object representing the physical system undergoing Langevin integration.
                      It should have attributes 'x' (position), 'v' (velocity), 'xi_m' (mass * friction coefficient), 
                      'sigma' (standard deviation of the random noise)
                      'dt' (time step) and 'h' (discretization interval for numerical force).
    - potential (object): An object representing the potential energy landscape of the system.
                         It should have a 'force' method that calculates the force at a given position.
    - bias (object or None, optional): An object representing the bias potential added to the of the system.
                         It should have a 'force' method that calculates the force at a given position.
    - eta_k (float or None, optional): If provided, use the given value as the random noise term (eta_k)
                        in the Langevin integrator. If None, draw a new value from a Gaussian normal distribution.
    - girsanov_reweighting (bool): If True Girsanov path reweighting factors are evaluated on-the-fly.

    Returns:
    None: The function modifies the 'x' and 'v' attributes of the provided system object.
    """    
    # if eta_k is not provided, draw eta_k from Gaussian normal distribution
    if eta_k is None:
        eta_k = np.random.normal()
        system.eta_k = eta_k
     
    # update position
    if bias is not None:
        system.bias_force = bias.force(system.x, system.h)[0] 
        force = potential.force(system.x, system.h)[0] + system.bias_force 
    else:
        force = potential.force(system.x, system.h)[0] 

    system.x = system.x + (force / system.xi_m ) * system.dt  +  system.sigma * np.sqrt(system.dt) * eta_k
    
    if girsanov_reweighting:
        system.delta_eta[0] = system.pre_factor * system.bias_force
        system.logM = system.eta[0] * system.delta_eta[0] + 0.5 * system.delta_eta[0] * system.delta_eta[0]
    
    return None  


#---------------------------------------------------------------------
#   B A S I C   L A N G E V I N   I N T E G R A T I O N   S T E P S
#---------------------------------------------------------------------

# A step
def A_step(system, half_step = False):
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
def B_step(system, potential, bias = None, half_step = False):
    """
    Perform a Langevin integration B-step for a given system.

    Parameters:
    - system (object): An object representing the physical system undergoing Langevin integration.
                      It should have attributes 'v' (velocity), 'm' (mass), and 'x' (position).
    - potential (object): An object representing the potential energy landscape of the system.
                         It should have a 'force' method that calculates the force at a given position.
    - bias (object or None, optional): An object representing the bias potential or pertubation potential
                                       added to the of the system. It should have a 'force' method that 
                                       calculates the force at a given position.
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
    if bias is not None:
        system.bias_force = bias.force(system.x, system.h)[0] 
 
    system.v = system.v + (1 / system.m) * dt * potential.force(system.x, system.h)[0] 
    
    return None 

# O_step
def O_step(system, step_index=0, half_step = False, eta_k = None):
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
    if eta_k is None:
        eta_k = np.random.normal()
        system.eta[step_index] = eta_k
    else:
        system.eta = eta_k # TODO HOW DO WE DEFINE SHAPE FOR INPUT?

    d = np.exp(- system.xi * dt)
    f_v = np.sqrt( R * system.T *  (1 / system.m)  * (1 - np.exp(-2 * system.xi * dt)) ) 

    system.v = d * system.v +  f_v * system.eta[step_index]

    return None

#----------------------------------------------------------------------------
#    L A N G E V I N   S P L I T T I N G   A L G O R I T H M S 
# with option for biased simulation and Girsanov reweighting 
#----------------------------------------------------------------------------

# ABO integrator
def ABO(system, potential, bias=None, eta_k = None, girsanov_reweighting=False):
    """
    Perform a full Langevin integration step consisting of A-step, B-step, and O-step.

    Parameters:
    - system (object): An object representing the physical system undergoing Langevin integration.
                      It should have attributes 'x' (position), 'v' (velocity), 'm' (mass), 'xi'
                      (friction coefficient), 'T' (temperature), and 'dt' (time step).
    - potential (object): An object representing the potential energy landscape of the system.
                         It should have a 'force' method that calculates the force at a given position.
    - bias (object or None, optional): An object representing the bias potential added to the system.
                         It should have a 'force' method that calculates the force at a given position.
    - eta_k (float or None, optional): If provided, use the given value as the random noise term (eta_k)
                        in the Langevin integrator. If None, draw a new value from a Gaussian normal distribution.
    - girsanov_reweighting (bool): If True Girsanov path reweighting factors are evaluated on-the-fly.

    Returns:
    None: The function modifies the 'x' and 'v' attributes of the provided system object in place.
    """    
    
    A_step(system)
    B_step(system, potential, bias) 
    O_step(system, eta_k = eta_k)

    if girsanov_reweighting:
        system.delta_eta[0] = system.d / system.f * system.dt * system.bias_force
        system.logM = system.eta[0] * system.delta_eta[0] + 0.5 * system.delta_eta[0] * system.delta_eta[0]
    
    return None   

# ABOBA integrator
def ABOBA(system, potential, bias=None, eta_k = None, girsanov_reweighting=False):
    """
    Perform a full Langevin integration step for the ABOBA algorithm

    Parameters:
    - system (object): An object representing the physical system undergoing Langevin integration.
                      It should have attributes 'x' (position), 'v' (velocity), 'm' (mass), 'xi'
                      (friction coefficient), 'T' (temperature), and 'dt' (time step).
    - potential (object): An object representing the potential energy landscape of the system.
                         It should have a 'force' method that calculates the force at a given position.
    - bias (object or None, optional): An object representing the bias potential added to the system.
                         It should have a 'force' method that calculates the force at a given position.
    - eta_k (float or None, optional): If provided, use the given value as the random noise term (eta_k)
                        in the Langevin integrator. If None, draw a new value from a Gaussian normal distribution.
    - girsanov_reweighting (bool): If True Girsanov path reweighting factors are evaluated on-the-fly.

    Returns:
    None: The function modifies the 'x' and 'v' attributes of the provided system object in place.
    """    
    
    A_step(system, half_step= True)
    B_step(system, potential, bias=bias, half_step = True) 
    O_step(system, eta_k = eta_k)    
    B_step(system, potential, bias=bias, half_step = True) 
    A_step(system, half_step = True)
    
    if girsanov_reweighting:
        system.delta_eta[0] = (system.d + 1) / system.f * system.dt / 2 * system.bias_force
        system.logM = system.eta[0] * system.delta_eta[0] + 0.5 * system.delta_eta[0] * system.delta_eta[0]
    
    return None   

# AOBOA integrator
def AOBOA(system, potential, bias=None, eta_k = None, girsanov_reweighting=False):
    """
    Perform a full Langevin integration step for the AOBOA algorithm

    Parameters:
    - system (object): An object representing the physical system undergoing Langevin integration.
                      It should have attributes 'x' (position), 'v' (velocity), 'm' (mass), 'xi'
                      (friction coefficient), 'T' (temperature), and 'dt' (time step).
    - potential (object): An object representing the potential energy landscape of the system.
                         It should have a 'force' method that calculates the force at a given position.
    - bias (object or None, optional): An object representing the bias potential added to the system.
                         It should have a 'force' method that calculates the force at a given position.
    - eta_k (float or None, optional): If provided, use the given value as the random noise term (eta_k)
                        in the Langevin integrator. If None, draw a new value from a Gaussian normal distribution.
    - girsanov_reweighting (bool): If True Girsanov path reweighting factors are evaluated on-the-fly.

    Returns:
    None: The function modifies the 'x' and 'v' attributes of the provided system object in place.
    """    

    A_step(system, half_step = True)
    O_step(system, half_step = True, eta_k = eta_k, step_index=0) 
    B_step(system, potential, bias) 
    O_step(system, half_step = True, eta_k = eta_k, step_index=1) 
    A_step(system, half_step = True)
    
    if girsanov_reweighting:
        eta_combined = system.d_prime * system.eta[0] + system.eta[1]
        system.delta_eta[0] = system.d_prime / system.f_prime * system.dt * system.bias_force
        system.logM = eta_combined * system.delta_eta[0] / (system.d_prime * system.d_prime + 1) + 0.5 * system.delta_eta[0] * system.delta_eta[0] / (system.d_prime * system.d_prime + 1)
    
    return None      

# BAOAB integrator
def BAOAB(system, potential, bias=None, eta_k = None):
    """
    Perform a full Langevin integration step for the BAOAB algorithm

    Parameters:
    - system (object): An object representing the physical system undergoing Langevin integration.
                      It should have attributes 'x' (position), 'v' (velocity), 'm' (mass), 'xi'
                      (friction coefficient), 'T' (temperature), and 'dt' (time step).
    - potential (object): An object representing the potential energy landscape of the system.
                         It should have a 'force' method that calculates the force at a given position.
    - bias (object or None, optional): An object representing the bias potential added to the system.
                         It should have a 'force' method that calculates the force at a given position.
    - eta_k (float or None, optional): If provided, use the given value as the random noise term (eta_k)
                        in the Langevin integrator. If None, draw a new value from a Gaussian normal distribution.

    Returns:
    None: The function modifies the 'x' and 'v' attributes of the provided system object in place.
    """    
    
    B_step(system, potential, bias, half_step = True) 
    A_step(system, half_step = True)
    O_step(system, eta_k = eta_k)    
    A_step(system, half_step = True)
    B_step(system, potential, bias, half_step = True) 
    
    return None   

# BOAOB integrator
def BOAOB(system, potential, bias=None, eta_k = None, girsanov_reweighting=False):
    """
    Perform a full Langevin integration step for the BOAOB algorithm

    Parameters:
    - system (object): An object representing the physical system undergoing Langevin integration.
                      It should have attributes 'x' (position), 'v' (velocity), 'm' (mass), 'xi'
                      (friction coefficient), 'T' (temperature), and 'dt' (time step).
    - potential (object): An object representing the potential energy landscape of the system.
                         It should have a 'force' method that calculates the force at a given position.
    - bias (object or None, optional): An object representing the bias potential added to the system.
                         It should have a 'force' method that calculates the force at a given position.
    - eta_k (float or None, optional): If provided, use the given value as the random noise term (eta_k)
                        in the Langevin integrator. If None, draw a new value from a Gaussian normal distribution.
    - girsanov_reweighting (bool): If True Girsanov path reweighting factors are evaluated on-the-fly.

    Returns:
    None: The function modifies the 'x' and 'v' attributes of the provided system object in place.
    """    
    
    bias_force_k = system.bias_force
    B_step(system, potential, bias=bias, half_step = True) 
    O_step(system, half_step = True, eta_k = eta_k, step_index=0)
    A_step(system)
    O_step(system, half_step = True, eta_k = eta_k, step_index=1) 
    B_step(system, potential, bias=bias, half_step = True)     
    
    if girsanov_reweighting:
        system.delta_eta[0] = system.d_prime / system.f_prime * system.dt / 2 * bias_force_k
        system.delta_eta[1] = 1 / system.f_prime * system.dt / 2 * system.bias_force
        logM_1 = system.eta[0] * system.delta_eta[0] + 0.5 * system.delta_eta[0] * system.delta_eta[0]
        logM_2 = system.eta[1] * system.delta_eta[1] + 0.5 * system.delta_eta[1] * system.delta_eta[1]
        system.logM = logM_1 + logM_2 
    
    return None  

# OBABO integrator
def OBABO(system, potential, bias=None, eta_k = None, girsanov_reweighting=False):
    """
    Perform a full Langevin integration step for the OBABO algorithm

    Parameters:
    - system (object): An object representing the physical system undergoing Langevin integration.
                      It should have attributes 'x' (position), 'v' (velocity), 'm' (mass), 'xi'
                      (friction coefficient), 'T' (temperature), and 'dt' (time step).
    - potential (object): An object representing the potential energy landscape of the system.
                         It should have a 'force' method that calculates the force at a given position.
    - bias (object or None, optional): An object representing the bias potential added to the system.
                         It should have a 'force' method that calculates the force at a given position.
    - eta_k (float or None, optional): If provided, use the given value as the random noise term (eta_k)
                        in the Langevin integrator. If None, draw a new value from a Gaussian normal distribution.
    - girsanov_reweighting (bool): If True Girsanov path reweighting factors are evaluated on-the-fly.

    Returns:
    None: The function modifies the 'x' and 'v' attributes of the provided system object in place.
    """    

    bias_force_k = system.bias_force
    O_step(system, half_step = True, eta_k = eta_k, step_index=0)   
    B_step(system, potential, bias, half_step = True) 
    A_step(system)
    B_step(system, potential, bias, half_step = True)
    O_step(system, half_step = True, eta_k = eta_k, step_index=1)   
    
    if girsanov_reweighting:
        system.delta_eta[0] = 1 / system.f_prime * system.dt / 2 * bias_force_k
        system.delta_eta[1] = system.d_prime / system.f_prime * system.dt / 2 * system.bias_force
        logM_1 = system.eta[0] * system.delta_eta[0] + 0.5 * system.delta_eta[0] * system.delta_eta[0]
        logM_2 = system.eta[1] * system.delta_eta[1] + 0.5 * system.delta_eta[1] * system.delta_eta[1]
        system.logM = logM_1 + logM_2 
    
    return None  

# OABAO integrator
def OABAO(system, potential, bias=None, eta_k = None):
    """
    Perform a full Langevin integration step for the OABAO algorithm

    Parameters:
    - system (object): An object representing the physical system undergoing Langevin integration.
                      It should have attributes 'x' (position), 'v' (velocity), 'm' (mass), 'xi'
                      (friction coefficient), 'T' (temperature), and 'dt' (time step).
    - potential (object): An object representing the potential energy landscape of the system.
                         It should have a 'force' method that calculates the force at a given position.
    - bias (object or None, optional): An object representing the bias potential added to the system.
                         It should have a 'force' method that calculates the force at a given position.
    - eta_k (float or None, optional): If provided, use the given value as the random noise term (eta_k)
                        in the Langevin integrator. If None, draw a new value from a Gaussian normal distribution.

    Returns:
    None: The function modifies the 'x' and 'v' attributes of the provided system object in place.
    """    
    
    O_step(system, half_step = True, eta_k = eta_k)   
    A_step(system, half_step = True)   
    B_step(system, potential, bias, half_step = True) 
    A_step(system, half_step = True)   
    O_step(system, half_step = True, eta_k = eta_k)   
    
    return None 

# BAOA integrator
def BAOA(system, potential, bias=None, eta_k = None):
    """
    Perform a full Langevin integration step for the BAOA algorithm

    Parameters:
    - system (object): An object representing the physical system undergoing Langevin integration.
                      It should have attributes 'x' (position), 'v' (velocity), 'm' (mass), 'xi'
                      (friction coefficient), 'T' (temperature), and 'dt' (time step).
    - potential (object): An object representing the potential energy landscape of the system.
                         It should have a 'force' method that calculates the force at a given position.
    - bias (object or None, optional): An object representing the bias potential added to the system.
                         It should have a 'force' method that calculates the force at a given position.
    - eta_k (float or None, optional): If provided, use the given value as the random noise term (eta_k)
                        in the Langevin integrator. If None, draw a new value from a Gaussian normal distribution.

    Returns:
    None: The function modifies the 'x' and 'v' attributes of the provided system object in place.
    """    
    
    B_step(system, potential, bias) 
    A_step(system, half_step = True)   
    O_step(system, eta_k = eta_k)   
    A_step(system, half_step = True)   
    
    return None 
