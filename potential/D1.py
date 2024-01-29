#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 07:05:58 2024

@author: bettina
"""

#-----------------------------------------
#   I M P O R T S 
#-----------------------------------------
from abc import ABC, abstractmethod
import numpy as np
import scipy.constants as const
from scipy import integrate
from scipy.optimize import minimize


#------------------------------------------------
# abstract class: one-dimensional potentials
#------------------------------------------------
class D1(ABC):
    #---------------------------------------------------------------------
    #   class initialization needs to be implemented in a child class
    #
    #   In the initialization define the parameters of the potential
    #   and the range [x_low, x_high]
    #---------------------------------------------------------------------
    @abstractmethod    
    def __init__(self, param): 
        pass
    #---------------------------------------------------------------------
    #   analyitical functions that need to be implemented in a child class
    #---------------------------------------------------------------------
    # the potential energy function 
    @abstractmethod
    def potential(self, x):
        pass
    
    # the force, analytical expression
    @abstractmethod
    def force(self, x):
        pass
    
    # the Hessian matrix, analytical expression
    @abstractmethod    
    def hessian(self, x):
        pass

    #-----------------------------------------------------------
    #   numerical methods that are passed to a child class
    #-----------------------------------------------------------
    # negated potential, returns - V(x)
    def negated_potential(self, x): 
        """
        Calculate the negated potential energy -V(x) 

        The units of V(x) are kJ/mol, following the convention in GROMACS.

        Parameters:
            - x (float): position

        Returns:
            float: negated value of the potential energy function at the given position x.
        """
        return -self.potential(x)    
    
    # force, numerical expression via finite difference    
    def force_num(self, x, h):
        """
        Calculate the force F(x) numerically via the central finit difference.
        Since the potential is one-idmensional, the force is vector with one element.
        
        The force is given by:
        F(x) = - [ V(x+h/2) - V(x-h/2)] / h
        
        The units of F(x) are kJ/(mol * nm), following the convention in GROMACS.  
        
        Parameters:
        - x (float): position
 
        Returns:
            numpy array: The value of the force at the given position x , returned as vector with 1 element.  
        """  
        
        F = - ( self.potential(x+h/2) - self.potential(x-h/2) ) / h
        return np.array([F])
    
    # Hessian matrix, numerical expreesion via second order finite difference
    def hessian_num(self, x, h):
        """
        Calculate the Hessian matrix H(x) numerically via the second-order central finit difference.
        Since the potential is one dimensional, the Hessian matrix has dimensions 1x1.
        
        The Hessian is given by:
            H(x) = [V(x+h) - 2 * V(x) + V(x-h)] / h**2
        
        The units of H(x) are kJ/(mol * nm * nm), following the convention in GROMACS.
        
        Parameters:
        - x (float): position
        - h (float): spacing of the finit different point along x
        
        Returns:
        numpy array: The 1x1 Hessian matrix at the given position x.
        
        """
        
        # calculate the Hessian as a float    
        V_x_plus_h = self.potential(x+h)
        V_x = self.potential(x)
        V_x_minus_h = self.potential(x-h)
        
        H = (V_x_plus_h - 2 * V_x + V_x_minus_h) / h**2
        
        # cast Hessian as a 1x1 numpy array and return
        return  np.array([[H]]) 
    
    # unnormalized Boltzmann factor
    def boltzmann_factor(self, x, T):
        """
        Calculate the unnormalized Boltzmann factor for the 1-dimensional Bolhuis potential 
        
        The unnormalized Boltzmann factor is given by:
        p(x) = exp(- V(x) * 1000 / (R * T))
        
        The potential is given in molar units (kJ/mol). Consequently, the ideal gas constant R is used rather than the Boltzmann constant k_B. 
        The factor 1000 arises from converting kJ/mol to J/mol
        
        Parameters:
            - T (float): temperature in units of K
              
        Returns:
            float: The value of the unnormalized Boltzmann factor at the given position x.
        """    
        
        return np.exp(-self.potential(x) * 1000 / (T * const.R))
        
    # partition function
    def partition_function(self, T, limits=None):
        """
        Calculate the partition function for the 1-dimensional Bolhuis potential 
        
        The partition function is given by:
        Q = \int_{-\infty}^{\infty} p(x) dx
        
        In practice, the integration bounds are set to specific values, specified in the variable limits.
        Integration is carried out by scipy.integrate
        
        Parameters:
            - T (float): temperatur in units of K
            - limits (list, optional): limits of he integrations. Defaults have been set at the initialization of the class 
                
        Returns:
            list: The value of the partition function and an error estimate [Q, Q_error] 
        
        """        
        
        try:         
            # no limits are passed, the class members x_low and x_high are available
            if limits==None and hasattr(self, 'x_low') and hasattr(self, 'x_high'):
                # integrate the unnormalized Boltzmann factor within the specified limits
                Q, Q_error = integrate.quad(self.boltzmann_factor, self.x_low, self.x_high,  args=(T))
                # return Q and the error estimates of the integation 
                return [Q, Q_error]
            
            # limits are passed, format is correct, ignore the class members x_low and x_high
            elif limits!=None and isinstance(limits, list) and len(limits) == 2:
                # integrate the unnormalized Boltzmann factor within the specified limits
                Q, Q_error = integrate.quad(self.boltzmann_factor, limits[0], limits[1],  args=(T))
                # return Q and the error estimates of the integation 
                return [Q, Q_error]
            
            # no limits are passed, but the class members x_low and x_high are missing
            elif limits==None and (not hasattr(self, 'x_low') or not hasattr(self, 'x_high') ):     
                raise ValueError("Default limits of the integration have not been implemented. Pass limits as an argument")
            
            # limit are passed, format is wrong, raise error
            elif limits!=None and (not isinstance(limits, list) or not len(limits) == 2):
                raise ValueError("Input 'limits' is not a list with two elements.")
            
        except Exception as e:
            print(f"Error: {e}")
            return False

    # transition state
    def TS(self, x_start, x_end):
        """
        Numerically finds the highest maximum in the interval [x_start, x_end] 
        
        Parameters:
        - x_start (float): position of the reactant minimum
        - x_start (float): position of the product minimum
        
        Returns:
        float: position of the transition state
        
        """
        
        # find the largest point in [x_start, x_end] on a grid        
        x = np.linspace(x_start, x_end, 1000)
        y = self.potential(x)
        i = np.argmax(y)
        # this is our starting point for the optimization
        TS_start = x[i]
        
        # minimize returns a class OptimizeResult
        # the transition state is the class member x
        TS = minimize(self.negated_potential, TS_start, method='BFGS').x
        
        # returns position of the transition state as float
        return TS[0]     


#------------------------------------------------
# child class: one-dimensional potentials
#------------------------------------------------
class Bolhuis(D1):
    # intiialize class
    def __init__(self, param): 
        """
        Initialize the class for the 1-dimensional Bolhuis potential based on the given parameters.

        Parameters:
            - param (list): a list of parameters representing:
            - param[0]: a (float) - parameter controlling the center of the quadratic term.
            - param[1]: b (float) - parameter controlling the width of the quadratic term.
            - param[2]: c (float) - parameter controlling the width of perturbation.
            - param[3]: k1 (float) - force constant of the double well. Default is 1.d
            - param[4]: k2 (float) - force constant of the linear term. Default is 0.
            - param[5]: alpha (float) - strength of the perturbation.


        For the partition function, we need to speficy an intervall  [x_low, x_high]. 
        This intervall is set to default values at the intialization, 
        but can other limits can be specified in the function call for the partition function. 
        
        x_low = a - 3 sqrt(|b|)
        x_high  = a + 3 sqrt(|b|)
    
        where |b| is the absolute value of b. The reason for this is that for k2=0 and alpha=0, the extrama of the potential are located at
        
        x_max = a
        x_min = a +/- sqrt(b)
        
        The the boundaries are located at three times the distance between minimum and maximum on either side of the maximum. 


        Raises:
        - ValueError: If param does not have exactly 6 elements.
        """
        
        # Check if param has the correct number of elements
        if len(param) != 6:
            raise ValueError("param must have exactly 6 elements.")
        
        # Assign parameters
        self.a = param[0]
        self.b = param[1]
        self.c = param[2]
        self.k1 = param[3]
        self.k2 = param[4]
        self.alpha = param[5]
        
        # calculate a likely range of the function 
        self.x_low = self.a-3*np.sqrt(np.abs(self.b))
        self.x_high = self.a+3*np.sqrt(np.abs(self.b))

    # the potential energy function 
    def potential(self, x):
        """
        Calculate the potential energy V(x) for the 1-dimensional Bolhuis potential.
    
        The potential energy function is given by:
        V(x) = k1 * ((x - a)**2 - b)**2 + k2 * x + alpha * np.exp(-c * (x - a)**2)
    
        The units of V(x) are kJ/mol, following the convention in GROMACS.
    
        Parameters:
            - x (float): position

        Returns:
            float: The value of the potential energy function at the given position x.
        """
    
        return  self.k1 * ((x - self.a)**2 - self.b)**2 + self.k2 * x + self.alpha * np.exp(-self.c * (x - self.a)**2)
          
    # the force, analytical expression t
    def force(self, x):
        """
        Calculate the force F(x) analytically for the 1-dimensional Bolhuis potential.
        Since the potential is one-idmensional, the force is a vector with one element.
    
        The force is given by:
        F(x) = - dV(x) / dx 
              = - 2 * k1 * ((x - a)**2 - b) * 2 * (x-a) - k2 + alpha * np.exp(-c * (x - a)**2) * c * 2 * (x - a)
    
        The units of F(x) are kJ/(mol * nm), following the convention in GROMACS.
    
        Parameters:
            - x (float): position
    
        Returns:
            numpy array: The value of the force at the given position x, returned as vector with 1 element.
    
        """
        
        F = - 2 * self.k1 * ((x - self.a)**2 - self.b) * 2 * (x - self.a) - self.k2 + self.alpha * np.exp(-self.c * (x - self.a)**2) * self.c * 2 * (x - self.a)
        return np.array([F])
    
    # the Hessian matrix, analytical expression
    def hessian(self, x):
          """
          Calculate the Hessian matrx H(x) analytically for the 1-dimensional Bolhuis potential.
          Since the potential is one dimensional, the Hessian matrix has dimensions 1x1.
    
          The Hessian is given by:
          H(x) = d^2 V(x) / dx^2 
                = 12 * k1 (x - a)**2   +   4 * k1 * b   +   2 * alpha * c * [ 4 * c * (x-a)**2 - (x-a)] * exp (-c *(x-2)**2 )
    
          The units of H(x) are kJ/(mol * nm * nm), following the convention in GROMACS.
    
        Parameters:
            - x (float): position

          Returns:
              numpy array: The 1x1 Hessian matrix at the given position x.
    
          """
          
          # calculate the Hessian as a float      
          H = 12 * self.k1 * (x - self.a)**2   -   4 * self.k1 * self.b   +   2 * self.alpha * self.c * ( 2 * self.c * (x-self.a)**2 - 1 ) * np.exp (-self.c *(x-self.a)**2 )
          
          # cast Hessian as a 1x1 numpy array and return
          return  np.array([[H]])
      
    


