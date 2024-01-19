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
from scipy import integrate

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
    
# the Hessian matrix, analytical expression
def H(x, a=2, b=1, c=20, k1=1, k2=0, alpha=0):
      """
      Calculate the Hessian matrx H(x) analytically for the 1-dimensional Bolhuis potential based on the given parameters.
      Since the potential is one dimensional, the Hessian matrix has dimensions 1x1.

      The Hessian is given by:
      H(x) = d^2 V(x) / dx^2 
           = 12 * k1 (x - a)**2   +   4 * k1 * b   +   2 * alpha * c * [ 4 * c * (x-a)**2 - (x-a)] * exp (-c *(x-2)**2 )

      The units of H(x) are kJ/(mol * nm * nm), following the convention in GROMACS.

      Parameters:
          - x (float): position
          - a (float, optional): parameter controlling the center of the quadratic term. Default is 2.
          - b (float, optional): parameter controlling the width of the quadratic term. Default is 1.
          - c (float, optional): parameter controlling the width of perturbation. Default is 20
          - k1 (float, optional): force constant of the double well. Default is 1.
          - k2 (float, optional): force constant of the linear term. Default is 0.
          - alpha (float, optional): strength of the perturbation. Default is 0.

      Returns:
          numpy array: The 1x1 Hessian matrix at the given position x.

      """
      
      # calculate the Hessian as a float      
      H = 12 * k1 * (x - a)**2   -   4 * k1 * b   +   2 * alpha * c * ( 2 * c * (x-a)**2 - 1 ) * np.exp (-c *(x-2)**2 )
      
      # cast Hessian as a 1x1 numpy array and return
      return  np.array([[H]])
  
# the Hessian matrix, analytical expression
def H_num(x, h, a=2, b=1, c=20, k1=1, k2=0, alpha=0):
    """
    Calculate the Hessian matrix H(x) for the 1-dimensional Bolhuis potential numerically via the second-order central finit difference.
    Since the potential is one dimensional, the Hessian matrix has dimensions 1x1.
    
    The Hessian is given by:
    H(x) = [V(x+h) - 2 * V(x) + V(x-h)] / h**2
    
    The units of H(x) are kJ/(mol * nm * nm), following the convention in GROMACS.
    
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
        numpy array: The 1x1 Hessian matrix at the given position x.
    
    """
    # calculate the Hessian as a float    
    V_x_plus_h = V(x+h, a, b, c, k1, k2, alpha)
    V_x = V(x, a, b, c, k1, k2, alpha)
    V_x_minus_h = V(x-h, a, b, c, k1, k2, alpha)
    
    H = (V_x_plus_h - 2 * V_x + V_x_minus_h) / h**2
    
    # cast Hessian as a 1x1 numpy array and return
    return  np.array([[H]]) 
  
# the unnormalized Boltzmann factor
def p(x, T, a=2, b=1, c=20, k1=1, k2=0, alpha=0):
    """
    Calculate the unnormalized Boltzmann factor for the 1-dimensional Bolhuis potential 

    The unnormalized Boltzmann factor is given by:
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

# the partiition function 
def Q(T, a=2, b=1, c=20, k1=1, k2=0, alpha=0, limits=None):
    """
    Calculate the partition function for the 1-dimensional Bolhuis potential 

    The partition function is given by:
    Q = \int_{-\infty}^{\infty} p(x) dx

    In practice, the integration bounds are set to specific values, specified in the variable limits.
    Integration is carried out by scipy.integrate

    Parameters:
        - T (float): temperatur in units of K
        - a (float, optional): parameter controlling the center of the quadratic term. Default is 2.
        - b (float, optional): parameter controlling the width of the quadratic term. Default is 1.
        - c (float, optional): parameter controlling the width of perturbation. Default is 20
        - k1 (float, optional): force constant of the double well. Default is 1.
        - k2 (float, optional): force constant of the linear term. Default is 0.
        - alpha (float, optional): strength of the perturbation. Default is 0.
        - limits (list, optional): limits of he integrations. Default is None. 
        
    If the limits are not passed as an argument they are set to
    
    x_low = a - 3 sqrt(|b|)
    x_high  = a + 3 sqrt(|b|)

    where |b| is the absolute value of b. The reason for this is that for k2=0 and alpha=0, the extrama of the potential are located at
    
    x_max = a
    x_min = a +/- sqrt(b)
    
    The the boundaries are located at three times the distance between minimum and maximum on either side of the maximum. 
        
    Returns:
        list: The value of the partition function and an error estimate [Q, Q_error] 

    """        
    
    try:         
        if limits==None: 
            # specify the limits
            limits = [a-3*np.sqrt(np.abs(b)), a+3*np.sqrt(np.abs(b))]
            # integrate the unnormalized Boltzmann factor within the specified limits
            Q, Q_error = integrate.quad(p, limits[0], limits[1],  args=(T, a, b, c, k1, k2, alpha))
            # return Q and the error estimates of the integation 
            return [Q, Q_error]
            
        elif limits!=None and isinstance(limits, list) and len(limits) == 2:
            # integrate the unnormalized Boltzmann factor within the specified limits
            Q, Q_error = integrate.quad(p, limits[0], limits[1],  args=(T, a, b, c, k1, k2, alpha))
            # return Q and the error estimates of the integation 
            return [Q, Q_error]

        else:
            raise ValueError("Input is not a list with two elements.")

    except Exception as e:
        print(f"Error: {e}")
        return False

