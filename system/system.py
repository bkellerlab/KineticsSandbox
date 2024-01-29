#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 11:04:33 2024

@author: bettina
"""

#------------------------------------------------
# class: one-dimensional potentials
#------------------------------------------------
class D1():
    # intiialize class
    def __init__(self, m, x, v, T, xi, dt): 
        """
        Initialize the class for a 1-dimensional system based on the given parameters.

        Parameters:
            - m (float): mass in units of u
            - x (float): position in units of nm
            - v (float): velocity in units of nm/ps
            - T (float): temperature in untis of K
            - xi (float): collision frequency in units of 1/ps
            - dt (float): time step in units of ps
        """
        
        # Assign parameters
        self.m = m
        self.x = x
        self.v = v
        self.T = T
        self.xi = xi
        self.dt = dt
