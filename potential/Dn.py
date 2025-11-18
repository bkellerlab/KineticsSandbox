#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 13:59:16 2025

@author: sascha
"""

# -----------------------------------------
#   I M P O R T S
# -----------------------------------------
from abc import ABC, abstractmethod
import numpy as np
# from scipy.optimize import minimize

from .D1 import Quadratic as _D1_Quadratic
# ------------------------------------------------
# abstract class: one-dimensional potentials
# ------------------------------------------------


class Dn(ABC):
    # ---------------------------------------------------------------------
    #   class initialization needs to be implemented in a child class
    #
    #   In the initialization define the parameters of the potential
    #   and the range [q_low, q_high]
    # ---------------------------------------------------------------------
    @abstractmethod
    def __init__(self, param):
        pass
    # ---------------------------------------------------------------------
    #   analytical functions that need to be implemented in a child class
    # ---------------------------------------------------------------------
    # the potential energy function

    @abstractmethod
    def potential(self, q):
        pass

    # the force, analytical expression
    @abstractmethod
    def force_ana(self, q):
        pass

    # # the Hessian matrix, analytical expression
    # @abstractmethod
    # def hessian_ana(self, q):
    #     pass

    # -----------------------------------------------------------
    #   numerical methods that are passed to a child class
    # -----------------------------------------------------------
    # negated potential, returns - V(q)
    def negated_potential(self, q):
        """
      Calculate the negated potential energy -V(q)

        The units of V(q) are kJ/mol, following the convention in GROMACS.

        Parameters:
            - q (ndarray): position vector

        Returns:
            float: negated value of the potential energy function at the given
                   position q.
        """
        return -self.potential(q)

    # force, numerical expression via finite difference
    def force_num(self, q, h):
        """
        Calculate the force F(q) numerically via the central finit difference.

        The force per dimension is given by:
        F_i(q) = - [ V_i(q+h/2) - V_i(q-h/2)] / h

        The units of F(q) are kJ/(mol * nm), following the convention in
        GROMACS.

        Parameters:
        - q (ndarray): position vector
        - h (float/ndarray): displacement (float: one for all dimensions)

        Returns:
            numpy array: The value of the force at the given position x ,
                         returned as vector with 1 element.
        """

        _h = np.ones_like(q) * h
        F = np.zeros_like(q)

        for _i in range(len(q)):
            q_p = q.copy()
            q_m = q.copy()
            q_p[_i] += _h[_i]/2
            q_m[_i] -= _h[_i]/2
            F[_i] += - (self.potential(q_p) - self.potential(q_m)) / _h[_i]

        return F

#     # Hessian matrix, numerical expreesion via second order finite difference
#     def hessian_num(self, x, h):
#         """
#         Calculate the Hessian matrix H(x) numerically via the second-order central finit difference.
#         Since the potential is one dimensional, the Hessian matrix has dimensions 1x1.
#         
#         The Hessian is given by:
#             H(x) = [V(x+h) - 2 * V(x) + V(x-h)] / h**2
#         
#         The units of H(x) are kJ/(mol * nm * nm), following the convention in GROMACS.
#         
#         Parameters:
#         - x (float): position
#         - h (float): spacing of the finit different point along x
#         
#         Returns:
#         numpy array: The 1x1 Hessian matrix at the given position x.
#         
#         """
#         
#         # calculate the Hessian as a float    
#         V_x_plus_h = self.potential(x+h)
#         V_x = self.potential(x)
#         V_x_minus_h = self.potential(x-h)
#         
#         H = (V_x_plus_h - 2 * V_x + V_x_minus_h) / h**2
#         
#         # cast Hessian as a 1x1 numpy array and return
#         return  np.array([[H]]) 

    # ---------------------------------------------------------------------------------
    #   functions that automatically switch between analytical and numerical function
    # ---------------------------------------------------------------------------------
    # for the force
    def force(self, x, h):
        # try whether the analytical force is implemented
        try:
            F = self.force_ana(x)
        # if force_ana(x) returns a NotImplementedError, use the numerical force instead
        except NotImplementedError:
            F = self.force_num(x, h)
        return F

#     # for the hessian
#     def hessian(self, x, h):
#         # try whether the analytical hessian is implemted
#         try:
#             H = self.hessian_ana(x)
#         # if hessian_ana(x) returns a NotImplementedError, use the numerical hessian instead
#         except NotImplementedError:
#             H = self.hessian_num(x, h)
#         return H


class Quadratic(Dn):
    # intiialize class
    def __init__(self, param):
        """
        Initialize the class for the n-dimensional quandratic potential based on the given parameters.

        Parameters:
            - param (list): a list of parameter tuples (or 2D numpy array) representing:
                [(k_0, a_0), (k_1, a_1), ..., (k_n, a_n)]
                with:

            - param[i][0]: k_i (float) - force constant of potential in
                                         dimension i
            - param[i][1]: a_i (float) - parameter that shifts the extremum
                                         left and right in dimension i

        Raises:
        - ValueError: If param[i] does not have exactly 2 elements.
        """

        # Check if param has the correct number of elements
        _param = np.array(param)
        if _param.shape[-1] != 2:
            raise ValueError("param[i] must have exactly 2 elements.")

        # Assign parameters
        self.k = _param[:, 0]
        self.a = _param[:, 1]

    # the potential energy function
    def potential(self, q):
        """
        Calculate the potential energy V(q) for the n-dimensional quadratic potential.

        The potential energy function per dimension is given by:
        V(x) = k * 0.5 * (x-a)**2

        The units of V(q) are kJ/mol, following the convention in GROMACS.

        Parameters:
            - q (ndarray): n-dimensional position vector

        Returns:
            float: The value of the potential energy function at the given
                   position q.
        """

        return  np.sum(self.k * 0.5 * (q - self.a)**2)

    # the force, analytical expression
    def force_ana(self, q):
        """
        Calculate the force F(q) analytically for the n-dimensional quadratic potential.

        The force per dimension is given by:
        F(x) = - dV(x) / dx
             = - k * (x-a)

        The units of F(q) are kJ/(mol * nm), following the convention in GROMACS.

        Parameters:
            - q (ndarray): n-dimensional position vector

        Returns:
            numpy array: The value of the force at the given position q,
            returned as vector with n elements.

        """

        F = -self.k * (q - self.a)
        return F
