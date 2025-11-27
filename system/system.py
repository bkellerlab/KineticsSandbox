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
class D1:
    def __init__(
        self,
        m: float,
        x: float,
        v: float,
        T: float,
        xi: float,
        dt: float,
        h: float,
        eta: np.ndarray = np.array([None, None]),
        delta_eta: np.ndarray = np.array([None, None]),
        bias_force: float | None = None,
        logM: float | None = None
    ):
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
            - eta (array-like, optional): random number(s) at integration step k.
                    e.g.: in BOAOB/OBABO/BP two random numbers are drawn `eta_1` and `eta_2`
            - delta_eta (array-like, optional): random number difference(s) according to 
                    J. Phys. Chem. B 2024, 128, 6014−6027, recorded if Girsanov_reweighting is True
            - bias_force (float, optional): bias potential force in units of kJ/mol, recorded if bias_force is True
            - logM (float, optional): path entropy, recorded if Girsanov_reweighting is True 
        """
        
        # Assign parameters
        self.m = m
        self.x = x
        self.v = v
        self.T = T
        self.xi = xi
        self.dt = dt
        self.h = h
        self.eta = eta
        self.delta_eta = delta_eta
        self.bias_force = bias_force
        self.logM = logM
        
        # get natural constants in the appropriate units    
        # the factor 0.001 converts R into the kJ / (mol * K)
        R = const.R * 0.001

        # calculate some combined parameters
        # mass-weighted friction
        self.xi_m = self.xi * self.m
        # diffusion constant
        self.D = self.T * R / self.xi_m
        # sigma = standard deviation of random noise in overdamped Langevin dynamics
        self.sigma = np.sqrt(2* self.D)

        # update parameters for Grisanov reweighting for overdamped Langevin dynamics
        self.pre_factor =  np.sqrt(self.dt / (2 * self.T * R * self.xi_m) )
        # update parameters for Grisanov reweighting for Langevin dynamics
        self.d = np.exp(- self.xi * self.dt)
        self.f = np.sqrt(R * self.T * self.m  * (1 - np.exp(-2 * self.xi * self.dt)))
        self.d_prime = np.exp(- self.xi * self.dt / 2)
        self.f_prime = np.sqrt(R * self.T *  self.m  * (1 - np.exp(-self.xi * self.dt)))

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