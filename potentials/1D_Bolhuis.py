#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 09:11:17 2024

@author: bettina
"""

def V(x, k2, k1, a):
    """
    Calculate the potential energy V(x) for the 1-dimensional Bolhuis potential based on the given parameters.

    Parameters:
        - x (float): position
        - k2 (float): force constant of the double well
        - k1 (float): force constant of the linear term
        - a (float): strength of the perturbation

    Returns:
        float: The computed potential energy.

    Formula:
        V(x) = k2 * ((x - 2)**2 - 1)**2 + k1 * x + a * np.exp(-20 * (x - 2)**2)

    Example:
        >>> V(3, 2, 1, 0.5)
        58.5
    """
    
    return  k2 * ((x - 2)**2 - 1)**2 + k1 * x + a * np.exp(-20 * (x - 2)**2)

