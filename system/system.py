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
    def __init__(self, m, x, v, T, xi, dt, h, eta_k=None, eta_l=None, bias_force=None, logM=None): 
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
            - eta_k (float): random number drwan for propagation 
                             Note: in BOAOB/ OBABO/BP == eta1; in AOBOA combined eta
            - ela_l (float): second random number for BOAOB/ OBABO/BP integrator
            - bias_force (func): function to describe bias potential force
            - logM (float): path entropy $1/D \Delta S$ in eq 28 J. Phys. Chem. B 2024, 128, 6014−6027 
        """
        
        # Assign parameters
        self.m = m
        self.x = x
        self.v = v
        self.T = T
        self.xi = xi
        self.dt = dt
        self.h = h
        self.eta_k = eta_k
        self.eta_l = eta_l
        self.bias_force = bias_force
        self.logM = logM
        
        # calculate some combined parameters
        # mass-weighted friction
        self.xi_m = self.xi * self.m
        # diffusion constant
        # the factor 0.001 converts R into the kJ / (mol * K)
        self.D = self.T * const.R * 0.001 / self.xi_m
        # sigma = standard deviation of random noise in overdamped Langevin dynamics
        self.sigma = np.sqrt(2* self.D)
        