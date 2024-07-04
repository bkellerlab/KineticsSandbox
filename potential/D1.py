#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 07:05:58 2024

@author: bettina
"""

# -----------------------------------------
#   I M P O R T S
# -----------------------------------------
from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import minimize

# -----------------------------------------
#   W R A P E R
# -----------------------------------------


def apply_nested_2d_array(func):
    def wrapper(self, x, *args, **kwargs):
        # Ensure x is a numpy array
        x = np.asarray(x)

        # Calculate the values using the original function
        values = func(self, x, *args, **kwargs)

        def to_nested_2d_array(value):
            if np.isscalar(value):
                return np.array([[value]])
            elif isinstance(value, np.ndarray):
                if value.ndim == 0:
                    return np.array([[value.item()]])
                else:
                    return np.array([to_nested_2d_array(v) for v in value])
            else:
                raise ValueError("Unhandled value type")

        # Apply the conversion to the entire values array
        nested_values = to_nested_2d_array(values)

        return nested_values

    return wrapper


def apply_nested_1d_array(func):
    def wrapper(self, x, *args, **kwargs):
        # Ensure x is a numpy array
        x = np.asarray(x)

        # Calculate the values using the original function
        values = func(self, x, *args, **kwargs)

        def to_nested_1d_array(value):
            if np.isscalar(value):
                return np.array([value])
            elif isinstance(value, np.ndarray):
                if value.ndim == 0:
                    return np.array([value.item()])
                else:
                    return np.array([to_nested_1d_array(v) for v in value])
            else:
                raise ValueError("Unhandled value type")

        # Apply the conversion to the entire values array
        nested_values = to_nested_1d_array(values)

        return nested_values

    return wrapper


# ------------------------------------------------
# abstract class: one-dimensional potentials
# ------------------------------------------------
class D1(ABC):
    # ---------------------------------------------------------------------
    #   class initialization needs to be implemented in a child class
    #
    #   In the initialization define the parameters of the potential
    #   and the range [x_low, x_high]
    # ---------------------------------------------------------------------
    @abstractmethod
    def __init__(self, param):
        pass

    # ---------------------------------------------------------------------
    #   analytical functions that need to be implemented in a child class
    # ---------------------------------------------------------------------
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

    # -----------------------------------------------------------
    #   numerical methods that are passed to a child class
    # -----------------------------------------------------------
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

    # -------------------- playground arthur: start ------------------------------------

    # ---------------------------------------------------------------------
    #   numerical differentiation methods
    # ---------------------------------------------------------------------

    # Primitive backward difference for the first derivative
    def primitive_backward_difference_first(self, x, h):
        """
        First-order accurate backward difference approximation for the first derivative.
        f_prime(x) ≈ [f(x) - f(x - h)] / h

        Parameters:
        - x (float or numpy array): position or positions
        - h (float): spacing of the finite difference points along x

        Returns:
        numpy array: The value of the first derivative at the given position(s) x, returned as a vector with n = len(x) elements.
        """
        x = np.asarray(x)
        f_prime = (self.potential(x) - self.potential(x - h)) / h
        return f_prime

    # Primitive backward difference for the second derivative
    def primitive_backward_difference_second(self, x, h):
        """
        First-order accurate backward difference approximation for the second derivative.
        f_double_prime(x) ≈ [f(x) - 2f(x - h) + f(x - 2h)] / h^2

        Parameters:
        - x (float or numpy array): position or positions
        - h (float): spacing of the finite difference points along x

        Returns:
        numpy array: The value of the second derivative at the given position(s) x, returned as a vector with n = len(x) elements.
        """
        x = np.asarray(x)
        f_double_prime = (
            self.potential(x) - 2 * self.potential(x - h) + self.potential(x - 2 * h)
        ) / h**2
        return f_double_prime

    # Primitive forward difference for the first derivative
    def primitive_forward_difference_first(self, x, h):
        """
        First-order accurate forward difference approximation for the first derivative.
        f_prime(x) ≈ [f(x + h) - f(x)] / h

        Parameters:
        - x (float or numpy array): position or positions
        - h (float): spacing of the finite difference points along x

        Returns:
        numpy array: The value of the first derivative at the given position(s) x, returned as a vector with n = len(x) elements.
        """
        x = np.asarray(x)
        f_prime = (self.potential(x + h) - self.potential(x)) / h
        return f_prime

    # Primitive forward difference for the second derivative
    def primitive_forward_difference_second(self, x, h):
        """
        First-order accurate forward difference approximation for the second derivative.
        f_double_prime(x) ≈ [f(x + 2h) - 2f(x + h) + f(x)] / h^2

        Parameters:
        - x (float or numpy array): position or positions
        - h (float): spacing of the finite difference points along x

        Returns:
        numpy array: The value of the second derivative at the given position(s) x, returned as a vector with n = len(x) elements.
        """
        x = np.asarray(x)
        f_double_prime = (
            self.potential(x + 2 * h) - 2 * self.potential(x + h) + self.potential(x)
        ) / h**2
        return f_double_prime

    # Primitive central difference for the first derivative
    def primitive_central_difference_first(self, x, h):
        """
        Second-order accurate central difference approximation for the first derivative.
        f_prime(x) ≈ - [ f(x + h/2) - f(x - h/2)] / h

        Parameters:
        - x (float or numpy array): position or positions
        - h (float): spacing of the finite difference points along x
        Returns:
        numpy array: The value of the first derivative at the given position(s) x, returned as a vector with n = len(x) elements.
        """
        x = np.asarray(x)
        f_prime = -(self.potential(x + h / 2) - self.potential(x - h / 2)) / h
        return f_prime

    # Primitive central difference for the second derivative
    def primitive_central_difference_second(self, x, h):
        """
        Second-order accurate central difference approximation for the second derivative.
        f_double_prime(x) ≈ [f(x + h) - 2f(x) + f(x - h)] / h^2
        Source: https://www.dam.brown.edu/people/alcyew/handouts/numdiff.pdf

        Parameters:
        - x (float or numpy array): position or positions
        - h (float): spacing of the finite difference points along x

        Returns:
        numpy array: The value of the second derivative at the given position(s) x, returned as a vector with n = len(x) elements.
        """
        x = np.asarray(x)
        f_double_prime = (
            self.potential(x + h) - 2 * self.potential(x) + self.potential(x - h)
        ) / h**2
        return f_double_prime

    # Forward difference for the first derivative
    def forward_difference_first(self, x, h):
        """
        Second-order accurate forward difference approximation for the first derivative.
        f_prime(x) ≈ [-3f(x) + 4f(x + h) - f(x + 2h)] / (2h)
        Source: https://www.dam.brown.edu/people/alcyew/handouts/numdiff.pdf

        Parameters:
        - x (float or numpy array): position or positions
        - h (float): spacing of the finite difference points along x

        Returns:
        numpy array: The value of the first derivative at the given position(s) x, returned as a vector with n = len(x) elements.
        """
        x = np.asarray(x)
        f_prime = (
            -3 * self.potential(x)
            + 4 * self.potential(x + h)
            - self.potential(x + 2 * h)
        ) / (2 * h)
        return f_prime

    # Forward difference for the second derivative
    def forward_difference_second(self, x, h):
        """
        Second-order accurate forward difference approximation for the second derivative.
        f_double_prime(x) ≈ [2f(x) - 5f(x + h) + 4f(x + 2h) - f(x + 3h)] / h^2
        Source: https://www.dam.brown.edu/people/alcyew/handouts/numdiff.pdf

        Parameters:
        - x (float or numpy array): position or positions
        - h (float): spacing of the finite difference points along x

        Returns:
        numpy array: The value of the second derivative at the given position(s) x, returned as a vector with n = len(x) elements.
        """
        x = np.asarray(x)
        f_double_prime = (
            2 * self.potential(x)
            - 5 * self.potential(x + h)
            + 4 * self.potential(x + 2 * h)
            - self.potential(x + 3 * h)
        ) / h**2
        return f_double_prime

    # Backward difference for the first derivative
    def backward_difference_first(self, x, h):
        """
        Second-order accurate backward difference approximation for the first derivative.
        f_prime(x) ≈ [3f(x) - 4f(x - h) + f(x - 2h)] / (2h)
        Source: https://www.dam.brown.edu/people/alcyew/handouts/numdiff.pdf

        Parameters:
        - x (float or numpy array): position or positions
        - h (float): spacing of the finite difference points along x

        Returns:
        numpy array: The value of the first derivative at the given position(s) x, returned as a vector with n = len(x) elements.
        """
        x = np.asarray(x)
        f_prime = (
            3 * self.potential(x)
            - 4 * self.potential(x - h)
            + self.potential(x - 2 * h)
        ) / (2 * h)
        return f_prime

    # Backward difference for the second derivative
    def backward_difference_second(self, x, h):
        """
        Second-order accurate backward difference approximation for the second derivative.
        f_double_prime(x) ≈ [2f(x) - 5f(x - h) + 4f(x - 2h) - f(x - 3h)] / h^2
        Source: https://www.dam.brown.edu/people/alcyew/handouts/numdiff.pdf

        Parameters:
        - x (float or numpy array): position or positions
        - h (float): spacing of the finite difference points along x

        Returns:
        numpy array: The value of the second derivative at the given position(s) x, returned as a vector with n = len(x) elements.
        """
        x = np.asarray(x)
        f_double_prime = (
            2 * self.potential(x)
            - 5 * self.potential(x - h)
            + 4 * self.potential(x - 2 * h)
            - self.potential(x - 3 * h)
        ) / h**2
        return f_double_prime

    # Higher-order central difference for the first derivative
    def central_difference_first(self, x, h):
        """
        Fourth-order accurate central difference approximation for the first derivative.
        f_prime(x) ≈ [8(f(x+h) - f(x-h)) - (f(x+2h) - f(x-2h))] / (12h)
        Source: https://www.dam.brown.edu/people/alcyew/handouts/numdiff.pdf

        Parameters:
        - x (float or numpy array): position or positions
        - h (float): spacing of the finite difference points along x

        Returns:
        numpy array: The value of the first derivative at the given position(s) x, returned as a vector with n = len(x) elements.
        """
        x = np.asarray(x)
        f_prime = (
            8 * (self.potential(x + h) - self.potential(x - h))
            - (self.potential(x + 2 * h) - self.potential(x - 2 * h))
        ) / (12 * h)
        return f_prime

    # Higher-order central difference for the second derivative
    def central_difference_second(self, x, h):
        """
        Fourth-order accurate central difference approximation for the second derivative.
        f_double_prime(x) ≈ [-f(x+2h) + 16f(x+h) - 30f(x) + 16f(x-h) - f(x-2h)] / (12h^2)
        Source: https://www.dam.brown.edu/people/alcyew/handouts/numdiff.pdf

        Parameters:
        - x (float or numpy array): position or positions
        - h (float): spacing of the finite difference points along x

        Returns:
        numpy array: The value of the second derivative at the given position(s) x, returned as a vector with n = len(x) elements.
        """
        x = np.asarray(x)
        V_x_plus_2h = self.potential(x + 2 * h)
        V_x_plus_h = self.potential(x + h)
        V_x = self.potential(x)
        V_x_minus_h = self.potential(x - h)
        V_x_minus_2h = self.potential(x - 2 * h)
        f_double_prime = (
            -V_x_plus_2h + 16 * V_x_plus_h - 30 * V_x + 16 * V_x_minus_h - V_x_minus_2h
        ) / (12 * h**2)

        return f_double_prime

    # force, numerical expression via higher-order finite difference
    def force_num(self, x, h, method=("central", 4)):
        """
        Calculate the force F(x) numerically via a higher-order central finite difference.
        Since the potential is one-dimensional, the force is a vector with one element.

        The force is given by ether:

        1. Fourth-order (4) accurate "central" difference approximation for the first derivative:
            f_prime(x) ≈ [ 8(f(x+h) - f(x-h)) - (f(x+2h) - f(x-2h)) ] / (12h)
            F = - self.central_difference_first(x, h)

        2. Second-order (2) accurate "central" difference approximation for the first derivative:
            f_prime(x) ≈ - [ f(x + h/2) - f(x - h/2) ] / h
            F = - self.primitive_central_difference_first(x, h)

        3. Second-order (2) accurate "forward" difference approximation for the first derivative:
            f_prime(x) ≈ [-3f(x) + 4f(x + h) - f(x + 2h)] / (2h)
            F = - self.forward_difference_first(x, h)

        4. First-order (1) accurate "forward" difference approximation for the first derivative:
            f_prime(x) ≈ [f(x + h) - f(x)] / h
            F = - self.primitive_forward_difference_first(x, h)

        5. Second-order (2) accurate "backward" difference approximation for the first derivative:
            f_prime(x) ≈ [3f(x) - 4f(x - h) + f(x - 2h)] / (2h)
            F = - self.backward_difference_first(x, h)

        6. First-order (1) accurate "backward" difference approximation for the first derivative:
            f_prime(x) ≈ [f(x) - f(x - h)] / h
            F = - self.primitive_backward_difference_first(x, h)

        The units of F(x) are kJ/(mol * nm), following the convention in GROMACS.

        Parameters:
        - x (float or numpy array): position(s)
        - h (float): spacing of the finite difference points along x
        - method=(type_of_finite_difference, order_of_accuracy), with:

            type_of_finite_difference   = "central", "backward" or "forward" as strings
            order_of_accuracy           = 1, 2 or 4 as intergers

        Returns:
        float, numpy array: The value(s) of the force at the given position(s) x, returned as float (if x is scalar) or numpy array (if x is an array) depending on the input structure of x.
        """
        if isinstance(x, list):
            x = np.asarray(x)
        else:
            pass

        if method[0] == "central":
            if method[1] == 4:
                F = -self.central_difference_first(x, h)
            elif method[1] == 2:
                F = -self.primitive_central_difference_first(x, h)
            else:
                print(
                    "Input Error: please provide method = ('central', 4) or ('central', 2)"
                )
        elif method[0] == "backward":
            if method[1] == 2:
                F = -self.backward_difference_first(x, h)
            elif method[1] == 1:
                F = -self.primitive_backward_difference_first(x, h)
            else:
                print(
                    "Input Error: please provide method = ('backward', 2) or ('backward', 1)"
                )
        elif method[0] == "forward":
            if method[1] == 2:
                F = -self.forward_difference_first(x, h)
            elif method[1] == 1:
                F = -self.primitive_forward_difference_first(x, h)
            else:
                print(
                    "Input Error: please provide method = ('forward', 2) or ('forward', 1)"
                )
        else:
            print(
                "Input Error: please provide method = ('forward', 2) or ('forward', 1) or ('backward', 2) or ('backward', 1) or ('central', 4) or ('central', 2)"
            )

        return F

    # Hessian matrix, numerical expression via higher-order finite difference
    def hessian_num(self, x, h, method=("central", 4)):
        """
        Calculate the Hessian matrix H(x) numerically via a central finite difference of n-th order accuracy.
        Since the potential is one-dimensional, the Hessian matrix has dimensions 1x1.

        The Hessian is given by ether:

        1. Fourth-order (4) accurate "central" difference approximation for the second derivative:
            f_double_prime(x) ≈ [-f(x+2h) + 16f(x+h) - 30f(x) + 16f(x-h) - f(x-2h)] / (12h^2)
            H = self.central_difference_second(x, h)

        2. Second-order (2) accurate "central" difference approximation for the second derivative:
            f_double_prime(x) ≈ [f(x + h) - 2f(x) + f(x - h)] / h^2
            H = self.primitive_central_difference_second(x, h)

        3. Second-order (2) accurate "forward" difference approximation for the second derivative:
            f_double_prime(x) ≈ [2f(x) - 5f(x + h) + 4f(x + 2h) - f(x + 3h)] / h^2
            H = self.forward_difference_second(x, h)

        4. First-order (1) accurate "forward" difference approximation for the second derivative:
            f_double_prime(x) ≈ [f(x + 2h) - 2f(x + h) + f(x)] / h^2
            H = self.primitive_forward_difference_second(x, h)

        5. Second-order (2) accurate "backward" difference approximation for the second derivative:
            f_double_prime(x) ≈ [2f(x) - 5f(x - h) + 4f(x - 2h) - f(x - 3h)] / h^2
            H = self.backward_difference_second(x, h)

        6. First-order (1) accurate "backward" difference approximation for the second derivative:
            f_double_prime(x) ≈ [f(x) - 2f(x - h) + f(x - 2h)] / h^2
            H = self.primitive_backward_difference_second(x, h)


        The units of H(x) are kJ/(mol * nm^2), following the convention in GROMACS.

        Parameters:
        - x (float or numpy array): position(s)
        - h (float): spacing of the finite difference points along x
        - method=(type_of_finite_difference, order_of_accuracy), with:

            type_of_finite_difference   = "central" or "backward" or "forward" as strings
            order_of_accuracy           = 1, 2 or 4 as intergers

        Returns:
            float, numpy array: The value(s) of the Hessian at the given position(s) x, returned as as float (if x is scalar) or numpy array (if x is an array) depending on the input structure of x.
        """

        if isinstance(x, list):
            x = np.asarray(x)
        else:
            pass

        # Calculate the Hessian as a float
        if method[0] == "central":
            if method[1] == 4:
                H = self.central_difference_second(x, h)
            elif method[1] == 2:
                H = self.primitive_central_difference_second(x, h)
            else:
                print(
                    "Input Error: please provide method = ('central', 4) or ('central', 2)"
                )
        elif method[0] == "backward":
            if method[1] == 2:
                H = self.backward_difference_second(x, h)
            elif method[1] == 1:
                H = self.primitive_backward_difference_second(x, h)
            else:
                print(
                    "Input Error: please provide method = ('backward', 2) or ('backward', 1)"
                )
        elif method[0] == "forward":
            if method[1] == 2:
                H = self.forward_difference_second(x, h)
            elif method[1] == 1:
                H = self.primitive_forward_difference_second(x, h)
            else:
                print(
                    "Input Error: please provide method = ('forward', 2) or ('forward', 1)"
                )
        else:
            print(
                "Input Error: please provide method = ('forward', 2) or ('forward', 1) or ('backward', 2) or ('backward', 1) or ('central', 4) or ('central', 2)"
            )

        # Cast Hessian as a 1x1 numpy array and return
        return H

    # -------------------- playground arthur: end ------------------------------------

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
        x_min = minimize(self.potential, x_start, method="BFGS").x

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
        TS = minimize(self.negated_potential, TS_start, method="BFGS").x

        # returns position of the transition state as float
        return TS[0]

    # ---------------------------------------------------------------------------------
    #   functions that automatically switch between analytical and numerical function
    # ---------------------------------------------------------------------------------
    # for the force
    def force(self, x, h, method=("central", 4)):
        # try whether the analytical force is implemted
        try:
            F = self.force_ana(x)
        # if force_ana(x) returns a NotImplementedError, use the numerical force instead
        except NotImplementedError:
            print("Analytical force not implemented, switching to numerical force.")
            F = self.force_num(x, h, method)
        return F

    # for the hessian
    def hessian(self, x, h, method=("central", 4)):
        # try whether the analytical hessian is implemted
        try:
            H = self.hessian_ana(x)
        # if hessian_ana(x) returns a NotImplementedError, use the numerical hessian instead
        except NotImplementedError:
            print("Analytical Hessian not implemented, switching to numerical Hessian.")
            H = self.hessian_num(x, h, method)
        return H


# ------------------------------------------------
# child class: one-dimensional potentials
# ------------------------------------------------


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

        return np.full_like(x, self.d)

    # the force, analytical expression
    def force_ana(self, x):
        """
        Calculate the force F(x) analytically for the 1-dimensional constant potential.
        Since the potential is one-didmensional, the force is a vector with one element.

        The force is given by:
        F(x) = - dV(x) / dx
              = 0

        The units of F(x) are kJ/(mol * nm), following the convention in GROMACS.

        Parameters:
            - x (float): position

        Returns:
            float, numpy array: The value(s) of the force at the given position(s) x, returned as float (if x is scalar) or numpy array (if x is an array) depending on the input structure of x.

        """

        F = np.full_like(x, 0)
        return F

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
              float, numpy array: The value(s) of the Hessian at the given position(s) x, returned as as float (if x is scalar) or numpy array (if x is an array) depending on the input structure of x.

        """

        # calculate the Hessian as a float
        H = np.full_like(x, 0)

        # cast Hessian as a 1x1 numpy array and return
        return H


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

        return self.k * (x - self.a)

    # the force, analytical expression
    def force_ana(self, x):
        """
        Calculate the force F(x) analytically for the 1-dimensional linear potential.
        Since the potential is one-didmensional, the force is a vector with one element.

        The force is given by:
        F(x) = - dV(x) / dx
              = - k

        The units of F(x) are kJ/(mol * nm), following the convention in GROMACS.

        Parameters:
            - x (float): position

        Returns:
            float, numpy array: The value(s) of the force at the given position(s) x, returned as float (if x is scalar) or numpy array (if x is an array) depending on the input structure of x.

        """

        F = np.full_like(x, -self.k)
        return F

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
              float, numpy array: The value(s) of the Hessian at the given position(s) x, returned as as float (if x is scalar) or numpy array (if x is an array) depending on the input structure of x..

        """

        # calculate the Hessian as a float
        H = np.full_like(x, 0)

        # cast Hessian as a 1x1 numpy array and return
        return H


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

        return self.k * 0.5 * (x - self.a) ** 2

    # the force, analytical expression
    def force_ana(self, x):
        """
        Calculate the force F(x) analytically for the 1-dimensional quadratic potential.
        Since the potential is one-didmensional, the force is a vector with one element.

        The force is given by:
        F(x) = - dV(x) / dx
              = - k * (x-a)

        The units of F(x) are kJ/(mol * nm), following the convention in GROMACS.

        Parameters:
            - x (float): position

        Returns:
            float, numpy array: The value(s) of the force at the given position(s) x, returned as float (if x is scalar) or numpy array (if x is an array) depending on the input structure of x.

        """

        F = -self.k * (x - self.a)
        return F

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
              float, numpy array: The value(s) of the Hessian at the given position(s) x, returned as as float (if x is scalar) or numpy array (if x is an array) depending on the input structure of x..

        """

        # calculate the Hessian as a float
        H = np.full_like(x, self.k)

        # cast Hessian as a 1x1 numpy array and return
        return H


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

        return self.k * ((x - self.a) ** 2 - self.b) ** 2

    # the force, analytical expression
    def force_ana(self, x):
        """
        Calculate the force F(x) analytically for the 1-dimensional double-well potential.
        Since the potential is one-didmensional, the force is a vector with one element.

        The force is given by:
        F(x) = - dV(x) / dx
              = - 4 * k * ((x-a)^2 - b) * (x-a)

        The units of F(x) are kJ/(mol * nm), following the convention in GROMACS.

        Parameters:
            - x (float): position

        Returns:
            float, numpy array: The value(s) of the force at the given position(s) x, returned as float (if x is scalar) or numpy array (if x is an array) depending on the input structure of x.

        """

        F = -4 * self.k * ((x - self.a) ** 2 - self.b) * (x - self.a)
        return F

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
              float, numpy array: The value(s) of the Hessian at the given position(s) x, returned as as float (if x is scalar) or numpy array (if x is an array) depending on the input structure of x..

        """

        # calculate the Hessian as a float
        H = 12 * self.k * (x - self.a) ** 2 - 4 * self.k * self.b

        # cast Hessian as a 1x1 numpy array and return
        return H


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
        self.a = param[0]
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

        return (
            self.c1 * (x - self.a)
            + self.c2 * (x - self.a) ** 2
            + self.c3 * (x - self.a) ** 3
            + self.c4 * (x - self.a) ** 4
            + self.c5 * (x - self.a) ** 5
            + self.c6 * (x - self.a) ** 6
        )

    # the force, analytical expression
    def force_ana(self, x):
        """
        Calculate the force F(x) analytically for the 1-dimensional polynomial potential.
        Since the potential is one-didmensional, the force is a vector with one element.

        The force is given by:
        F(x) = - dV(x) / dx
              = - c1 - 2*c2*(x-a) - 3*c3*(x-a)**2 - 4*c4*(x-a)**3 - 5*c5*(x-a)**4 - 6*c6*(x-a)**5

        The units of F(x) are kJ/(mol * nm), following the convention in GROMACS.

        Parameters:
            - x (float): position

        Returns:
            float, numpy array: The value(s) of the force at the given position(s) x, returned as float (if x is scalar) or numpy array (if x is an array) depending on the input structure of x.

        """

        F = (
            -self.c1
            - 2 * self.c2 * (x - self.a)
            - 3 * self.c3 * (x - self.a) ** 2
            - 4 * self.c4 * (x - self.a) ** 3
            - 5 * self.c5 * (x - self.a) ** 4
            - 6 * self.c6 * (x - self.a) ** 5
        )
        return F

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
              float, numpy array: The value(s) of the Hessian at the given position(s) x, returned as as float (if x is scalar) or numpy array (if x is an array) depending on the input structure of x..

        """

        # calculate the Hessian as a float
        H = (
            2 * self.c2
            + 6 * self.c3 * (x - self.a)
            + 12 * self.c4 * (x - self.a) ** 2
            + 20 * self.c5 * (x - self.a) ** 3
            + 30 * self.c6 * (x - self.a) ** 4
        )

        # cast Hessian as a 1x1 numpy array and return
        return H


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

        return (
            self.k1 * ((x - self.a) ** 2 - self.b) ** 2
            + self.k2 * x
            + self.alpha * np.exp(-self.c * (x - self.a) ** 2)
        )

    # the force, analytical expression
    def force_ana(self, x):
        """
        Calculate the force F(x) analytically for the 1-dimensional Bolhuis potential.
        Since the potential is one-didmensional, the force is a vector with one element.

        The force is given by:
        F(x) = - dV(x) / dx
              = - 2 * k1 * ((x - a)**2 - b) * 2 * (x-a) - k2 + alpha * np.exp(-c * (x - a)**2) * c * 2 * (x - a)

        The units of F(x) are kJ/(mol * nm), following the convention in GROMACS.

        Parameters:
            - x (float): position

        Returns:
            float, numpy array: The value(s) of the force at the given position(s) x, returned as float (if x is scalar) or numpy array (if x is an array) depending on the input structure of x.

        """

        F = (
            -2 * self.k1 * ((x - self.a) ** 2 - self.b) * 2 * (x - self.a)
            - self.k2
            + self.alpha
            * np.exp(-self.c * (x - self.a) ** 2)
            * self.c
            * 2
            * (x - self.a)
        )
        return F

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
              float, numpy array: The value(s) of the Hessian at the given position(s) x, returned as as float (if x is scalar) or numpy array (if x is an array) depending on the input structure of x..

        """

        # calculate the Hessian as a float
        H = (
            12 * self.k1 * (x - self.a) ** 2
            - 4 * self.k1 * self.b
            + 2
            * self.alpha
            * self.c
            * (2 * self.c * (x - self.a) ** 2 - 1)
            * np.exp(-self.c * (x - self.a) ** 2)
        )

        # cast Hessian as a 1x1 numpy array and return
        return H


class Prinz(D1):
    # intiialize class
    def __init__(self):
        """
        Initialize the class for the 1-dimensional Prinz potential.
        All parameters are hard-coded.
        """

    # the potential energy function
    def potential(self, x):
        """
        Calculate the potential energy V(x) for the 1-dimensional constant potential.

        The potential energy function is given by:
        V(x) = 4* ( x^8 + 0.8 * e^(-80x^2) + 0.2 * e^(-80(x-0.5)^2) + 0.5 * e^(-40(x+0.5)^2) )

        The units of V(x) are kJ/mol, following the convention in GROMACS.

        Parameters:
            - x (float): position

        Returns:
            float: The value of the potential energy function at the given position x.
        """

        return 4 * (
            x**8
            + 0.8 * np.exp(-80 * x**2)
            + 0.2 * np.exp(-80 * (x - 0.5) ** 2)
            + 0.5 * np.exp(-40 * (x + 0.5) ** 2)
        )

    # the force, analytical expression
    def force_ana(self, x):
        """
        The class method force_ana(x) is not implemented in the class for the Prinz potential.
        Use force_num(x,h) instead.
        """

        raise NotImplementedError(
            "potential.D1(Prinz) does not implement force_ana(self, x)"
        )

    # the Hessian matrix, analytical expression
    def hessian_ana(self, x):
        """
        The class method hessian_ana(x) is not implemented in the class for the Prinz potential.
        Use hessian_num(x,h) instead.
        """

        raise NotImplementedError(
            "potential.D1(Prinz) does not implement hessian_ana(self, x)"
        )


class Morse(D1):
    def __init__(self, param):
        """
        Initialize the Morse potential class with the given parameters.

        Parameters:
            - param (list): a list of parameters representing:
                - param[0]: D_e (float) - Well depth of the potential.
                - param[1]: a (float) - Width parameter of the potential.
                - param[2]: x_e (float) - Equilibrium bond distance.

        Raises:
            - ValueError: If param does not have exactly 3 elements.
        """
        if len(param) != 3:
            raise ValueError("param must have exactly 3 elements.")

        self.D_e = param[0]
        self.a = param[1]
        self.x_e = param[2]

    def potential(self, x):
        """
        Calculate the Morse potential V(x).

        The potential energy function is given by:
         V(x) = D_e (1 - e^{(-a(x - x_e))})^2

        Parameters:
            - x (float): position

        Returns:
            float: The value of the Morse potential at the given position x.
        """
        V = self.D_e * (1 - np.exp(-self.a * (x - self.x_e))) ** 2
        return V

    def force_ana(self, x):
        """
        Calculate the force F(x) analytically from the Morse potential.

        The force is given by:
        F(x) = - dV(x) / dx
             = -2aD_e (exp(-a(x - x_e)) - exp(-2a(x - x_e)))

        Parameters:
            - x (float): position

        Returns:
            numpy array: The value of the force at the given position x, returned as a vector with 1 element.
        """

        F = (
            -2
            * self.a
            * self.D_e
            * (np.exp(-self.a * (x - self.x_e)) - np.exp(-2 * self.a * (x - self.x_e)))
        )

        return F

    def hessian_ana(self, x):
        """
        Calculate the Hessian H(x) analytically from the Morse potential.

        The Hessian is given by:
        H(x) = d^2 V(x) / dx^2
             = 2a^2D_e (2e^(-2a(x - x_e)) - e^(-a(x - x_e)))

        Parameters:
            - x (float): position

        Returns:
            float, numpy array: The value(s) of the Hessian at the given position(s) x, returned as as float (if x is scalar) or numpy array (if x is an array) depending on the input structure of x.
        """

        H = (
            2
            * self.a**2
            * self.D_e
            * (
                2 * np.exp(-2 * self.a * (x - self.x_e))
                - np.exp(-self.a * (x - self.x_e))
            )
        )

        return H
