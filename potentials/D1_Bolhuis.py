#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 09:11:17 2024

@author: bettina
"""

#-----------------------------------------
#   I M P O R T S 
#-----------------------------------------
import numpy as np
import scipy.constants as const


# the potential energy function 
def V(x, a=2, b=1, c=20, k1=1, k2=0, alpha=0):
    """
    Calculate the potential energy V(x) for the 1-dimensional Bolhuis potential based on the given parameters.

    The potential energy function is given by:
    V(x) = k1 * ((x - a)**2 - b)**2 + k2 * x + alpha * np.exp(-c * (x - a)**2)

    The units of V(x) are kJ/mol, following the convention in GROMACS.

    Parameters:
        - x (float): position
        - a (float, optional): parameter controlling the center of the quadratic term. Default is 2.
        - b (float, optional): parameter controlling the width of the quadratic term. Default is 1.
        - c (float, optional): parameter controlling the width of perturbation. Default is 20
        - k1 (float, optional): force constant of the double well. Default is 1.
        - k2 (float, optional): force constant of the linear term. Default is 0.
        - alpha (float, optional): strength of the perturbation. Default is 0.

    Hint: 
        The width of the perturbation c should be as broad to as the maximum of the double well. 
        Plot the potential to make sure that the effect of the perturbationn is as intended.

    Returns:
        float: The value of the potential energy function at the given position x.

    Example:
        >>> V(4, 5, 20, 3, 2, 1, 0.5)
        369.0
    """
    
    return  k1 * ((x - a)**2 - b)**2 + k2 * x + alpha * np.exp(-c * (x - a)**2)

# the force, analytical expression
def F(x, a=2, b=1, c=20, k1=1, k2=0, alpha=0):
    """
    Calculate the force F(x) analytically for the 1-dimensional Bolhuis potential based on the given parameters.

    The force is given by:
    F(x) = - dV(x) / dx 
         = - 2 * k1 * ((x - a)**2 - b) * 2 * (x-a) - k2 + alpha * np.exp(-c * (x - 2)**2) * c * 2 * (x - 2)

    The units of F(x) are kJ/(mol * nm), following the convention in GROMACS.

    Parameters:
        - x (float): position
        - a (float, optional): parameter controlling the center of the quadratic term. Default is 2.
        - b (float, optional): parameter controlling the width of the quadratic term. Default is 1.
        - c (float, optional): parameter controlling the width of perturbation. Default is 20
        - k1 (float, optional): force constant of the double well. Default is 1.
        - k2 (float, optional): force constant of the linear term. Default is 0.
        - alpha (float, optional): strength of the perturbation. Default is 0.

    Returns:
        float: The value of the force at the given position x.

    """
    
    return  - 2 * k1 * ((x - a)**2 - b) * 2 * (x-a) - k2 + alpha * np.exp(-c * (x - 2)**2) * c * 2 * (x - 2)

# the force, numerical expression via finite difference
def F_num(x, h, a=2, b=1, c=20, k1=1, k2=0, alpha=0):
    """
    Calculate the force F(x) for the 1-dimensional Bolhuis potential numerically via the central finit difference

    The force is given by:
    F(x) = - [ V(x+h/2) - V(x-h/2)] / h

    The units of F(x) are kJ/(mol * nm), following the convention in GROMACS.

    Parameters:
        - x (float): position
        - h (float): spacing of the finit different point along x
        - a (float, optional): parameter controlling the center of the quadratic term. Default is 2.
        - b (float, optional): parameter controlling the width of the quadratic term. Default is 1.
        - c (float, optional): parameter controlling the width of perturbation. Default is 20
        - k1 (float, optional): force constant of the double well. Default is 1.
        - k2 (float, optional): force constant of the linear term. Default is 0.
        - alpha (float, optional): strength of the perturbation. Default is 0.

    Returns:
        float: The value of the force at the given position x.

    """    
    return - (V(x+h/2, a, b, c, k1, k2, alpha) - V(x-h/2, a, b, c, k1, k2, alpha)) / h
  
# the unnormalized Boltzmann factor
def p(x, T, a=2, b=1, c=20, k1=1, k2=0, alpha=0):
    """
    Calculate the unnormalized Boltzmann factor for the 1-dimensional Bolhuis potential 

    The unnormalized Boltzmann facto is given by:
    p(x) = exp(- V(x) * 1000 / (R * T))

    The potential is given in molar units (kJ/mol). Consequently, the ideal gas constant R is used rather than the Boltzmann constant k_B. 
    The factor 1000 arises from converting kJ/mol to J/mol

    Parameters:
        - x (float): position
        - T (float): temperatur in units of K
        - a (float, optional): parameter controlling the center of the quadratic term. Default is 2.
        - b (float, optional): parameter controlling the width of the quadratic term. Default is 1.
        - c (float, optional): parameter controlling the width of perturbation. Default is 20
        - k1 (float, optional): force constant of the double well. Default is 1.
        - k2 (float, optional): force constant of the linear term. Default is 0.
        - alpha (float, optional): strength of the perturbation. Default is 0.

    Returns:
        float: The value of the  unnormalized Boltzmann factor at the given position x.

    """    
    
    return np.exp(-V(x, a, b, c, k1, k2, alpha) * 1000 / (T * const.R))


