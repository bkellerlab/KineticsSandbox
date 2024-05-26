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
    #   analytical functions that need to be implemented in a child class
    #---------------------------------------------------------------------
    # the potential energy function 
    @abstractmethod
    def potential(self, x):
        pass
    
    # the force, analytical expression
    @abstractmethod
    def force_ana(self, x):
        pass
    
    # the Hessian matrix, analytical expression
    @abstractmethod    
    def hessian_ana(self, x):
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
    
    # nearest minimum
    def min(self, x_start): 
        """
        Numerically finds the nearest minimum in the vicinity of x_start 
        
        Parameters:
        - x_start (float): start of the minimization
        
        Returns:
        float: position of the minimum
        
        """        

        # This is a convenience function.
        # It essentially calls scipy.optimize.minimize.

        # minimize returns a class OptimizeResult
        # the minimum is the class member x
        x_min = minimize(self.potential, x_start, method='BFGS').x
        
        # returns position of the minimum as float
        return x_min[0]     

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
      
class Constant(D1):
    # intiialize class
    def __init__(self, param): 
        """
        Initialize the class for the 1-dimensional constant potential based on the given parameter.

        Parameters:
            - param (list): a list of parameters representing:
            - param[0]: d (float) - constant offset
  
        Raises:
        - ValueError: If param does not have exactly 1 element.
        """
        
        # Check if param has the correct number of elements
        if len(param) != 1:
            raise ValueError("param must have exactly 1 element.")
        
        # Assign parameters
        self.d = param[0]
        
    # the potential energy function 
    def potential(self, x):
        """
        Calculate the potential energy V(x) for the 1-dimensional constant potential.
    
        The potential energy function is given by:
        V(x) = d
    
        The units of V(x) are kJ/mol, following the convention in GROMACS.
    
        Parameters:
            - x (float): position

        Returns:
            float: The value of the potential energy function at the given position x.
        """
    
        return  np.full_like(x, self.d)
          
    # the force, analytical expression 
    def force_ana(self, x):
        """
        Calculate the force F(x) analytically for the 1-dimensional constant potential.
        Since the potential is one-idmensional, the force is a vector with one element.
    
        The force is given by:
        F(x) = - dV(x) / dx 
              = 0
    
        The units of F(x) are kJ/(mol * nm), following the convention in GROMACS.
    
        Parameters:
            - x (float): position
    
        Returns:
            numpy array: The value of the force at the given position x, returned as vector with 1 element.
    
        """
        
        F = np.full_like(x, 0)
        return np.array([F])
    
    # the Hessian matrix, analytical expression
    def hessian_ana(self, x):
          """
          Calculate the Hessian matrx H(x) analytically for the 1-dimensional constant potential.
          Since the potential is one dimensional, the Hessian matrix has dimensions 1x1.
    
          The Hessian is given by:
          H(x) = d^2 V(x) / dx^2 
                = 0
    
          The units of H(x) are kJ/(mol * nm * nm), following the convention in GROMACS.
    
        Parameters:
            - x (float): position

          Returns:
              numpy array: The 1x1 Hessian matrix at the given position x.
    
          """
          
          # calculate the Hessian as a float      
          H = np.full_like(x, 0)
          
          # cast Hessian as a 1x1 numpy array and return
          return  np.array([[H]]) 
      
class Linear(D1):
    # intiialize class
    def __init__(self, param): 
        """
        Initialize the class for the 1-dimensional linear potential based on the given parameters.

        Parameters:
            - param (list): a list of parameters representing:
            - param[0]: k (float) - force constant 
            - param[1]: a (float) - parameter that shifts the extremum left and right

        Raises:
        - ValueError: If param does not have exactly 2 elements.
        """
        
        # Check if param has the correct number of elements
        if len(param) != 2:
            raise ValueError("param must have exactly 2 elements.")
        
        # Assign parameters
        self.k = param[0]
        self.a = param[1]
        
    # the potential energy function 
    def potential(self, x):
        """
        Calculate the potential energy V(x) for the 1-dimensional linear potential.
    
        The potential energy function is given by:
        V(x) = k * (x - a) 
    
        The units of V(x) are kJ/mol, following the convention in GROMACS.
    
        Parameters:
            - x (float): position

        Returns:
            float: The value of the potential energy function at the given position x.
        """
    
        return  self.k * (x -self.a ) 
          
    # the force, analytical expression 
    def force_ana(self, x):
        """
        Calculate the force F(x) analytically for the 1-dimensional linear potential.
        Since the potential is one-idmensional, the force is a vector with one element.
    
        The force is given by:
        F(x) = - dV(x) / dx 
              = - k
    
        The units of F(x) are kJ/(mol * nm), following the convention in GROMACS.
    
        Parameters:
            - x (float): position
    
        Returns:
            numpy array: The value of the force at the given position x, returned as vector with 1 element.
    
        """
        
        F = np.full_like(x, -self.k)
        return np.array([F])
    
    # the Hessian matrix, analytical expression
    def hessian_ana(self, x):
          """
          Calculate the Hessian matrx H(x) analytically for the 1-dimensional linear potential.
          Since the potential is one dimensional, the Hessian matrix has dimensions 1x1.
    
          The Hessian is given by:
          H(x) = d^2 V(x) / dx^2 
                = 0
    
          The units of H(x) are kJ/(mol * nm * nm), following the convention in GROMACS.
    
        Parameters:
            - x (float): position

          Returns:
              numpy array: The 1x1 Hessian matrix at the given position x.
    
          """
          
          # calculate the Hessian as a float      
          H = np.full_like(x, 0)
          
          # cast Hessian as a 1x1 numpy array and return
          return  np.array([[H]])

class Quadratic(D1):
    # intiialize class
    def __init__(self, param): 
        """
        Initialize the class for the 1-dimensional quandratic potential based on the given parameters.

        Parameters:
            - param (list): a list of parameters representing:
            - param[0]: k (float) - force constant 
            - param[1]: a (float) - parameter that shifts the extremum left and right

        Raises:
        - ValueError: If param does not have exactly 2 elements.
        """
        
        # Check if param has the correct number of elements
        if len(param) != 2:
            raise ValueError("param must have exactly 2 elements.")
        
        # Assign parameters
        self.k = param[0]
        self.a = param[1]
        
    # the potential energy function 
    def potential(self, x):
        """
        Calculate the potential energy V(x) for the 1-dimensional quadratic potential.
    
        The potential energy function is given by:
        V(x) = k * 0.5 * (x-a)**2
    
        The units of V(x) are kJ/mol, following the convention in GROMACS.
    
        Parameters:
            - x (float): position

        Returns:
            float: The value of the potential energy function at the given position x.
        """
    
        return  self.k * 0.5 * (x - self.a)**2 
          
    # the force, analytical expression 
    def force_ana(self, x):
        """
        Calculate the force F(x) analytically for the 1-dimensional quadratic potential.
        Since the potential is one-idmensional, the force is a vector with one element.
    
        The force is given by:
        F(x) = - dV(x) / dx 
              = - k * (x-a)
    
        The units of F(x) are kJ/(mol * nm), following the convention in GROMACS.
    
        Parameters:
            - x (float): position
    
        Returns:
            numpy array: The value of the force at the given position x, returned as vector with 1 element.
    
        """
        
        F = -self.k * (x - self.a)
        return np.array([F])
    
    # the Hessian matrix, analytical expression
    def hessian_ana(self, x):
          """
          Calculate the Hessian matrx H(x) analytically for the 1-dimensional quadratic potential.
          Since the potential is one dimensional, the Hessian matrix has dimensions 1x1.
    
          The Hessian is given by:
          H(x) = d^2 V(x) / dx^2 
                = 2k
    
          The units of H(x) are kJ/(mol * nm * nm), following the convention in GROMACS.
    
        Parameters:
            - x (float): position

          Returns:
              numpy array: The 1x1 Hessian matrix at the given position x.
    
          """
          
          # calculate the Hessian as a float      
          H = np.full_like(x, self.k)
          
          # cast Hessian as a 1x1 numpy array and return
          return  np.array([[H]])

class DoubleWell(D1):
    # intiialize class
    def __init__(self, param): 
        """
        Initialize the class for the 1-dimensional double-well potential based on the given parameters.

        Parameters:
            - param (list): a list of parameters representing:
            - param[0]: k (float) - prefactor that scales the potential
            - param[1]: a (float) - parameter that shifts the extremum left and right
            - param[2]: b (float) - parameter controls the separation of the two wells

        Raises:
        - ValueError: If param does not have exactly 3 elements.
        """
        
        # Check if param has the correct number of elements
        if len(param) != 3:
            raise ValueError("param must have exactly 3 elements.")
        
        # Assign parameters
        self.k = param[0]
        self.a = param[1]
        self.b = param[2]

        
    # the potential energy function 
    def potential(self, x):
        """
        Calculate the potential energy V(x) for the 1-dimensional double-well potential.
    
        The potential energy function is given by:
        V(x) = k * ((x-a)**2 - b)**2
    
        The units of V(x) are kJ/mol, following the convention in GROMACS.
    
        Parameters:
            - x (float): position

        Returns:
            float: The value of the potential energy function at the given position x.
        """
    
        return  self.k * ((x - self.a)**2 - self.b)**2 
          
    # the force, analytical expression 
    def force_ana(self, x):
        """
        Calculate the force F(x) analytically for the 1-dimensional double-well potential.
        Since the potential is one-idmensional, the force is a vector with one element.
    
        The force is given by:
        F(x) = - dV(x) / dx 
              = - 4 * k * ((x-a)^2 - b) * (x-a)
    
        The units of F(x) are kJ/(mol * nm), following the convention in GROMACS.
    
        Parameters:
            - x (float): position
    
        Returns:
            numpy array: The value of the force at the given position x, returned as vector with 1 element.
    
        """
        
        F = -4 * self.k * ((x - self.a)**2 - self.b) * (x- self.a)
        return np.array([F])
    
    # the Hessian matrix, analytical expression
    def hessian_ana(self, x):
          """
          Calculate the Hessian matrx H(x) analytically for the 1-dimensional double-well potential.
          Since the potential is one dimensional, the Hessian matrix has dimensions 1x1.
    
          The Hessian is given by:
          H(x) = d^2 V(x) / dx^2 
               = 12 * k * (x-a)^2 - 4 * k * b
    
          The units of H(x) are kJ/(mol * nm * nm), following the convention in GROMACS.
    
        Parameters:
            - x (float): position

          Returns:
              numpy array: The 1x1 Hessian matrix at the given position x.
    
          """
          
          # calculate the Hessian as a float      
          H = 12 * self.k * (x - self.a)**2 - 4 * self.k * self.b
          
          # cast Hessian as a 1x1 numpy array and return
          return  np.array([[H]])
      
class Polynomial(D1):
    # intiialize class
    def __init__(self, param): 
        """
        Initialize the class for the 1-dimensional polynomial potential (up to order  6) based on the given parameters.

        Parameters:
            - param (list): a list of parameters representing:
            - param[0]: a (float) - parameter that shifts the extremum left and right                
            - param[1]: c1 (float) - parameter for term of order 1
            - param[2]: c2 (float) - parameter for term of order 2
            - param[3]: c3 (float) - parameter for term of order 3
            - param[4]: c4 (float) - parameter for term of order 4
            - param[5]: c5 (float) - parameter for term of order 5
            - param[6]: c6 (float) - parameter for term of order 6            


        Raises:
        - ValueError: If param does not have exactly 7 elements.
        """
        
        # Check if param has the correct number of elements
        if len(param) != 7:
            raise ValueError("param must have exactly 7 elements.")
        
        # Assign parameters
        self.a  = param[0]
        self.c1 = param[1]
        self.c2 = param[2]
        self.c3 = param[3]
        self.c4 = param[4]
        self.c5 = param[5]
        self.c6 = param[6]


    # the potential energy function 
    def potential(self, x):
        """
        Calculate the potential energy V(x) for the 1-dimensional polynomial potential.
    
        The potential energy function is given by:
        V(x) = c1*(x-a) + c2*(x-a)**2 + c3*(x-a)**3 + c4*(x-a)**4 + c5*(x-a)**5 + c6*(x-a)**6 
    
        The units of V(x) are kJ/mol, following the convention in GROMACS.
    
        Parameters:
            - x (float): position

        Returns:
            float: The value of the potential energy function at the given position x.
        """
    
        return  self.c1 * (x - self.a) + self.c2 * (x - self.a)**2 + self.c3 * (x - self.a)**3 + self.c4 * (x - self.a)**4 + self.c5 * (x - self.a)**5 + self.c6 * (x - self.a)**6
          
    # the force, analytical expression 
    def force_ana(self, x):
        """
        Calculate the force F(x) analytically for the 1-dimensional polynomial potential.
        Since the potential is one-idmensional, the force is a vector with one element.
    
        The force is given by:
        F(x) = - dV(x) / dx 
              = - c1 - 2*c2*(x-a) - 3*c3*(x-a)**2 - 4*c4*(x-a)**3 - 5*c5*(x-a)**4 - 6*c6*(x-a)**5 
    
        The units of F(x) are kJ/(mol * nm), following the convention in GROMACS.
    
        Parameters:
            - x (float): position
    
        Returns:
            numpy array: The value of the force at the given position x, returned as vector with 1 element.
    
        """
        
        F = - self.c1 - 2*self.c2*(x - self.a) - 3*self.c3*(x - self.a)**2 - 4*self.c4*(x - self.a)**3 - 5*self.c5*(x - self.a)**4 - 6*self.c6*(x - self.a)**5 
        return np.array([F])
    
    # the Hessian matrix, analytical expression
    def hessian_ana(self, x):
          """
          Calculate the Hessian matrx H(x) analytically for the 1-dimensional polynomial potential.
          Since the potential is one dimensional, the Hessian matrix has dimensions 1x1.
    
          The Hessian is given by:
          H(x) = d^2 V(x) / dx^2 
               = 2*c2 + 6*c3*(x-a)+ 12*c4*(x-a)**2 + 20*c5*(x-a)**3 + 30*c6*(x-a)**4 
    
          The units of H(x) are kJ/(mol * nm * nm), following the convention in GROMACS.
    
        Parameters:
            - x (float): position

          Returns:
              numpy array: The 1x1 Hessian matrix at the given position x.
    
          """
          
          # calculate the Hessian as a float      
          H = 2*self.c2 + 6*self.c3*(x - self.a)+ 12*self.c4*(x - self.a)**2 + 20*self.c5*(x - self.a)**3 + 30*self.c6*(x - self.a)**4
          
          # cast Hessian as a 1x1 numpy array and return
          return  np.array([[H]])

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
          
    # the force, analytical expression 
    def force_ana(self, x):
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
    def hessian_ana(self, x):
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
     
      