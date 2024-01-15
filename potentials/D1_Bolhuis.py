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


def V(x, a=2, b=1, c=20, k1=1, k2=0, alpha=0):
    """
    Calculate the potential energy V(x) for the 1-dimensional Bolhuis potential based on the given parameters.

    The potential energy function is given by:
    V(x) = k1 * ((x - a)**2 - b)**2 + k2 * x + alpha * np.exp(-c * (x - 2)**2)


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
    
    return  k1 * ((x - a)**2 - b)**2 + k2 * x + alpha * np.exp(-c * (x - 2)**2)

