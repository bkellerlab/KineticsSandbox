#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 11:04:33 2024

@author: bettina
"""
#-----------------------------------------
#   I M P O R T S 
#-----------------------------------------
import numpy as np
import scipy.constants as const

#------------------------------------------------
# class: one-dimensional potentials
#------------------------------------------------
class D1():
    # intiialize class
    def __init__(self, m, x, v, T, xi, dt, h): 
        """
        Initialize the class for a 1-dimensional system based on the given parameters.

        Parameters:
            - m (float): mass in units of u
            - x (float): position in units of nm
            - v (float): velocity in units of nm/ps
            - T (float): temperature in untis of K
            - xi (float): collision frequency in units of 1/ps
            - dt (float): time step in units of ps
            - h (float): discretization interval for numerical forces and hessian
        """
        
        # Assign parameters
        self.m = m
        self.x = x
        self.v = v
        self.T = T
        self.xi = xi
        self.dt = dt
        self.h = h
        
        # calculate some combined parameters
        # mass-weighted friction
        self.xi_m = self.xi * self.m
        # diffusion constant
        # the factor 0.001 converts R into the kJ / (mol * K)
        self.D = self.T * const.R * 0.001 / self.xi_m
        # sigma = standard deviation of random noise in overdamped Langevin dynamics
        self.sigma = np.sqrt(2* self.D)


#------------------------------------------------
# class: N-dimensional systems
"""
Created on Sat May 17 16:07:56 2025

@author: Ahmet Sarigun
"""
#------------------------------------------------
class Dn():
    # intiialize class
    def __init__(self, m, x, v, T, xi, dt, h): 
        """
        Initialize the class for an N-dimensional system based on the given parameters.

        Parameters:
            - m (float or array): mass in units of u (scalar or array broadcastable to x/v)
            - x (array): position vector in units of nm
            - v (array): velocity vector in units of nm/ps
            - T (float): temperature in units of K
            - xi (float): collision frequency in units of 1/ps
            - dt (float): time step in units of ps
            - h (float): discretization interval for numerical forces and hessian
        """
        
        # Convert to numpy arrays and ensure they have the same shape
        self.x = np.array(x, dtype=float)
        self.v = np.array(v, dtype=float)
        
        # Ensure x and v have the same shape
        if self.x.shape != self.v.shape:
            raise ValueError("Position and velocity arrays must have the same shape")
        
        # Assign parameters
        self.m = m
        self.T = T
        self.xi = xi
        self.dt = dt
        self.h = h
        
        # calculate some combined parameters
        # mass-weighted friction
        self.xi_m = self.xi * self.m
        # diffusion constant
        # the factor 0.001 converts R into the kJ / (mol * K)
        self.D = self.T * const.R * 0.001 / self.xi_m
        # sigma = standard deviation of random noise in overdamped Langevin dynamics
        self.sigma = np.sqrt(2* self.D)
        