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
#import scipy.constants as const
#from scipy import integrate
#from scipy.optimize import minimize
import matplotlib.pyplot as plt


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
    def force_num(self, x, h=0.0001):
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
        return np.array([F]).flatten() # use .flatten() for shape match
    
    # Hessian matrix, numerical expreesion via second order finite difference
    def hessian_num(self, x, h=0.0001):
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
        return  np.array([[H]]).flatten()
    
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
   
    
    # plotting
    def plot_function(self,x_values):
        """
        plot the potential function over a given range of x values
        """


        y_values = self.potential(x_values)
        dy_force = self.force(x_values)
        dy_force_num= self.force_num(x_values)
        dy_hessian = self.hessian(x_values)
        dy_hessian_num = self.hessian_num(x_values)


        plt.plot(x_values, y_values, label="f(x)", color="blue", linewidth=2, marker=".", markerfacecolor="k",
              markersize=4)

        plt.plot(x_values, dy_force, label="f'(x) - force", color="red", linewidth=2, marker=".", markerfacecolor="k",
              markersize=4)

        plt.plot(x_values, dy_force_num, label="f'(x) - force_num", color="green", linewidth=2, marker=".",
              markerfacecolor="k",
              markersize=4)

        plt.plot(x_values, dy_hessian, label="f''(x) - hessian", color="yellow", linewidth=2, marker=".",
              markerfacecolor="k",
              markersize=4)

        plt.plot(x_values, dy_hessian_num, label="f''(x) - hessian_num", color="grey", linewidth=2, marker=".",
              markerfacecolor="k",
              markersize=4)


        plt.title(f"{self.__class__.__name__}  and Derivatives Plot")

        plt.xlabel("x")
        plt.ylabel("f(x)/f'(x)/f''(x)")
        plt.legend()
        plt.grid()
        plt.savefig(f"{self.__class__.__name__}_and_Derivatives_fig.pdf")
        plt.show()




    # ------------------------------------------------

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

#---------------------------------------------
# child class: one-dimensional potentials
#------------------------------------------------
class Linear_Potential(D1):
    # initialize class
    def __init__(self, param):
         """
         Initialize the class for the 1-dimensional Linear potential based on the given parameters.
         parameters:
           - param (list): a list of parameters representing:
           - param[0]: c (float) -parameter controlling the steepness of the line
           - param[1]: m (float) _parameter giving the y-intercept
         """
         #assign parameters
         self.c = param[0]
         self.m = param[1]
    def potential(self, x):
        """
        calculate the linear potential energy V(x),
        The function is given by:
        V(x) = m * x + c
        parameters:

            x:position

        Returns:linear potential for all x

        """

        return self.m * x + self.c

    def force(self, x):
        """
        Calculate the force F(x) analytically for Linear potential
        The force is given by:
        F(x) = - dV(x) / dx
             = - m
        parameters:
                x: position

        Returns:numpy array: The value of the force at the given position x

        """
        if isinstance(x, float) or isinstance(x, int):
            return -1 * self.m
        else:
            # you are allowed to input x as an array, in this case the return is an array the same shape as x
            return np.full(x.shape, -1 * self.m)



    def hessian(self, x):

        if isinstance(x, float) or isinstance(x, int):
            return 0
        else:
            # you are allowed to input x as an array, in this case the return is an array the same shape as x
            return np.full(x.shape, 0)

    def riemann(self, a, b, n):
        """
        calculate the Riemann integral of the potential energy function over the interval [a,b)
        using riemann central method
        parameters:
                a(float): lower limit of integration
                b(float):upper limit of integration
                h(float):step size or width  of rectangle
                s(float):midpoint of interval
        return:
           area under the curve
        """
        h = (b - a) / n
        area = 0
        for i in range(n):
            s = a + (i + 0.5) * h
            area += self.potential(s) * h

        return area

#------------------------------------------------------
# child class: one-dimensional potentials
#------------------------------------------------------

class Quadratic_Potential(D1):

    def __init__(self, param):

        """
        Initialize the class for the 1-dimensional Quadratic potential based on the given parameters.
        parameters:
             - param (list): a list of parameters representing:
             - param[0]: a (float) - controlling width of parabola
             - param[1]: b (float) - controlling horizontal displacement of parabola
             - param[2]: c (float) - controlling the vertical displacement if parabola

        """
        #assign parameters
        self.a = param[0]
        self.b = param[1]
        self.c = param[2]



    def potential(self, x):
        """
        calculate the quadratic potential energy V(x),
        The function is given by:
        V(x) = a * x ** 2 * b * x + c
        parameters:
                x:position

        Returns:Quadratic potential for all x

        """



        return self.a * x**2 + self.b * x + self.c

    def force(self,x):
        """
        Calculate the force F(x) analytically for Quadratic potential
            The force is given by:
            F(x) = - dV(x) / dx
                 = - (2 * a * x + b)
            parameters:
                    x: position

        Returns:numpy array: The value of the force at the given position x

        """

        return -1 * (self.a * 2 * x + self.b)

    def hessian(self, x):
        """
        Calculate the Hessian matrx H(x) analytically for the 1-dimensional Quadratic potential.
        Since the potential is one dimensional, the Hessian matrix has dimensions 1x1.

            The Hessian is given by:
            H(x) = d^2 V(x) / dx^2
                 = 2a

            Parameters:
                - x (float): position

        Returns:
                numpy array: The 1x1 Hessian matrix at the given position x.

        """

        # calculate the Hessian as a float
        if isinstance(x, float) or isinstance(x, int):
            return 2 * self.a
        else:
            # you are allowed to input x as an array, in this case the return is an array the same shape as x
            return np.full(x.shape, 2 * self.a)


    def riemann(self, a, b, n):
        """
        calculate the Riemann integral of the potential energy function over the interval [a,b)
        using riemann central method
        parameters:
                a(float): lower limit of integration
                b(float):upper limit of integration
                h(float):step size or width  of rectangle
                s(float):midpoint of interval
        return:
           area under the curve
        """
        h = (b - a) / n
        area = 0
        for i in range(n):
            s = a + (i + 0.5) * h
            area += self.potential(s) * h

        return area

#-----------------------------------------------
# child class: one-dimensional potentials
#-----------------------------------------------
class Double_Well_Potential:
    pass


class Double_Well_Potential(D1):



    def __init__(self,param):
        """
        Initialize the class for the 1-dimensional Double well potential based on the given parameters.
        parameters:
          - param (list): a list of parameters representing:
          - param[0]: a (float) - controlling steepness of the wells
          - param[1]: b (float) - controlling  width and height of barrier between the wells
          - param[2]: c (float) - controlling the vertical shift of potential

        """

       #assign parameters
        self.a = param[0]
        self.b = param[1]
        self.c = param[2]


    def potential(self, x):
        """

        calculate the Double Well potential energy V(x),
        The function is given by:
        V(x) = a * x^4 - b * x^2 + c
        parameters:
                x:position

        Returns:Double well potential for all x
        """


        return  self.a * x**4 - self.b * x**2 + self.c

    def force(self,x):
        """
        Calculate the force F(x) analytically for Double Well  potential
        The force is given by:
        F(x) = - dV(x) / dx
             = - (4 * a * x ^ 3 - 2 * b * x)
        parameters:
             x: position
        Returns:numpy array: The value of the force at the given position x

        """


        return -1 * (self.a * 4 * x ** 3 - self.b * 2 * x)

    def hessian(self, x):
        """
        Calculate the Hessian matrx H(x) analytically for the 1-dimensional Quadratic potential.
        Since the potential is one dimensional, the Hessian matrix has dimensions 1x1.

         The Hessian is given by:
         H(x) = d^2 V(x) / dx^2
              =

         Parameters:
             - x (float): position

        Returns:
            numpy array: The 1x1 Hessian matrix at the given position x.

        """
        return 12 * x **2 * self.a - 2 * self.b

    def riemann(self, a, b, n):
        """
        calculate the Riemann integral of the potential energy function over the interval [a,b)
        using riemann central method
        parameters:
                    a(float): lower limit of integration
                    b(float):upper limit of integration
                    h(float):step size or width  of rectangle
                    s(float):midpoint of interval
        return:
               area under the curve
        """
        h = (b - a) / n
        area = 0
        for i in range(n):
            s = a + (i + 0.5) * h
            area += self.potential(s) * h

        return area



linear = Linear_Potential((2, -3))  # linear potential with m=2,c=-3
linear.plot_function(np.linspace(-10, 10, 100))
quadratic = Quadratic_Potential((1, -5, 6))  # Quadratic potential with a=1, b=-5, c=6
quadratic.plot_function(np.linspace(-10, 15, 100))
doublewell = Double_Well_Potential((1, 2, -1))  # double well potential with a=1,b=2,c=-1
doublewell.plot_function(np.linspace(-1.5, 1.5, 100))


        # ----------------------------




