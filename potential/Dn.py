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

from .D1 import D1
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


class PairPotentialSum(Dn):

    def __init__(
        self,
        ndim: int,
        natoms: int,
        pair_potentials: list[D1],
        pair_indices: np.ndarray[int],
    ):
        """
        Initialize the class for the Cartesian 3n-dimensional potential based
        on the sum of given D1 potentials with the input of atom pair distances

        Parameters:
            - ndim (int): Number of Cartesian coordinates (3 for xyz)
            - natoms (int): Number of atoms
            - pair_potentials (D1 shape (npairs)):
                List of D1 potential instances to sum over.
            - pair_indices (lists of integers, shape (npairs, 2)):
                Lists of atom pair indices.
        """

        # Check number of dimensions and particles input
        self.ndim = int(ndim)
        self.natoms = int(natoms)

        # Assign pair potential and particle pair indices
        self.pair_potentials = list(pair_potentials)
        if len(self.pair_potentials):
            self.pair_indices = np.array(pair_indices, dtype=int).reshape(
                2,
                len(self.pair_potentials)
            )
        else:
            self.pair_indices = np.array([[], []], dtype=int)

        return

    def add_potentials(
        self,
        pair_potentials: list[D1],
        pair_indices: np.ndarray[int],
    ):
        """
        Add a list of pair potentials and atom pair indices
        
        Parameters:
            - pair_potentials (D1 shape (npairs)):
                List of D1 potential instances to add.
            - pair_indices (lists of integers, shape (npairs, 2)):
                Lists of atom pair indices to add.
        """

        # Assign pair potential and particle pair indices
        self.pair_potentials += list(pair_potentials)
        self.pair_indices = np.concatenate(
            (
                self.pair_indices,
                np.array(pair_indices, dtype=int)
            ),
            axis=1
        )

        return

    def remove_potentials(
        self
    ):
        """
        Remove all pair potentials and atom pair indices
        """
        
        self.pair_potentials = []
        self.pair_indices = np.array([[], []], dtype=int)
        
        return

    def potential(
        self,
        x: np.ndarray,
    ) -> float:
        """
        Calculate the sum of pair potential energies V_i(q) for each declared
        atom pair.

        The units of V(q) are kJ/mol, following the convention in GROMACS.

        Parameters:
            - x (ndarray): (n_dim*n_atoms)-dimensional position vector

        Returns:
            float: 
                The value of the potential energy function at given atom
                positions x.

        """

        # Reshape positions to number of particles and dimensions
        xn = x.reshape(self.natoms, self.ndim).astype(float)

        # Compute particle pair distances
        distances = np.sqrt(
            np.sum(
                (
                    xn[self.pair_indices[0]]
                    - xn[self.pair_indices[1]]
                )**2,
                axis=1
            )
        )

        # Compute nD potential as sum of 1D potentials
        energy = np.sum(
            [
                pair_potential.potential(distance)
                for pair_potential, distance in zip(
                    self.pair_potentials, distances
                )
            ]
        )

        return energy

    def force_ana(
        self,
        x: np.ndarray,
    ) -> float:
        """
        Calculate the force F(q) analytically for the sum of pair potentials.

        The units of F(q) are kJ/(mol * nm), following the convention in
        GROMACS.

        Parameters:
            - x (ndarray): (n_dim*n_atoms)-dimensional position vector

        Returns:
            numpy array:
                The values of the force at the given position x returned as
                vector with (n_dim*n_atoms) elements.

        """

        # Reshape positions to number of particles and dimensions
        xn = x.reshape(self.natoms, self.ndim).astype(float)

        # Compute particle pair distances
        vectors = xn[self.pair_indices[0]] - xn[self.pair_indices[1]]
        distances = np.sqrt(
            np.sum(
                vectors**2,
                axis=1,
                keepdims=True,
            )
        )

        # Compute partial derivative of d distance / d positions
        drdx1 = vectors/distances
        drdx2 = -drdx1
        
        # Compute 1D forces F = - d V / d distance
        forces_distance = np.array(
            [
                pair_potential.force_ana(distance)
                for pair_potential, distance in zip(
                    self.pair_potentials, distances.reshape(-1)
                )
            ]
        )

        # Combine partial derivatives 
        # F(x) = - d V / d x = - d V / d distance * d distance /d x
        forces = np.zeros_like(xn)
        np.add.at(
            forces,
            self.pair_indices[0],
            forces_distance.reshape(-1, 1)*drdx1
        )
        np.add.at(
            forces,
            self.pair_indices[1],
            forces_distance.reshape(-1, 1)*drdx2
        )

        return forces.reshape(-1)


class BiasedPairPotentialSum(Dn):

    def __init__(
        self,
        pair_potential: Dn.Dn,
        bias_potential: Dn.Dn,
    ):
        """
        Initialize the class for the Cartesian 3n-dimensional potential based
        on the sum of given D1 potentials with the input of atom pair distances

        Parameters:
            pair_potential (Dn.Dn):
                Potential instance
            bias_potential (Dn.Dn):
                Bias potential instance

        """

        # Assign potential instance
        self.pair_potential = pair_potential

        # Assign bias potential instance
        self.bias_potential = bias_potential

        return

    def add_potentials(
        self,
        pair_potentials: list[D1],
        pair_indices: np.ndarray[int],
    ):
        """
        Add a list of pair potentials and atom pair indices
        
        Parameters:
            - pair_potentials (D1 shape (npairs)):
                List of D1 potential instances to add.
            - pair_indices (lists of integers, shape (npairs, 2)):
                Lists of atom pair indices to add.
        """

        # Assign pair potential and particle pair indices
        self.pair_potential.add_potentials(
            pair_potentials,
            pair_indices,
        )

        return

    def add_bias(
        self,
        bias_potentials: list[D1],
        bias_indices: np.ndarray[int],
    ):
        """
        Add a list of bias pair potentials and atom pair indices
        
        Parameters:
            - bias_potentials (D1 shape (npairs)):
                List of D1 bias potential instances to add.
            - bias_indices (lists of integers, shape (npairs, 2)):
                Lists of bias atom pair indices to add.
        """

        # Assign bias potential and particle pair indices
        self.bias_potential.add_potentials(
            bias_potentials,
            bias_indices,
        )

        return

    def remove_bias(
        self,
    ):
        """
        Remove all bias pair potentials and atom pair indices
        """

        # Assign bias potential and particle pair indices
        self.bias_potential.remove_potentials()

        return

    def potential(
        self,
        x: np.ndarray,
        verbose: bool = False
    ) -> float | List[float]:
        """
        Calculate the sum of pair potential energies V_i(q) for each declared
        atom pair.

        The units of V(q) are kJ/mol, following the convention in GROMACS.

        Parameters:
            - x (ndarray): (n_dim*n_atoms)-dimensional position vector
            - verbose (bool, optional):
                If False, only return sum of potential and bias. If True,
                return also the sum but also the bias, separately.

        Returns:
            float: 
                The value of the potential energy function and bias potential
                at given atom positions x.
            or (verbose = True)
            list(float):
                Both, the sum of the potential energy function and bias
                potential at given atom positions x, and the bias potential
                value, separately.

        """

        # Compute pair potential
        pair_energy = self.pair_potential.potential(x)

        # Compute bias potential
        bias_energy = self.bias_potential.potential(x)

        # Return just the potential sum or additionally the bias potential
        if verbose:
            return pair_energy + bias_energy, bias_energy
        else:
            return pair_energy + bias_energy

    def force_ana(
        self,
        x: np.ndarray,
    ) -> float:
        """
        Calculate the force F(q) analytically for the sum of pair potentials.

        The units of F(q) are kJ/(mol * nm), following the convention in
        GROMACS.

        Parameters:
            - x (ndarray): (n_dim*n_atoms)-dimensional position vector

        Returns:
            numpy array:
                The values of the force at the given position x returned as
                vector with (n_dim*n_atoms) elements.

        """


        # Compute pair potential forces
        pair_forces = self.pair_potential.force_ana(x)
        
        # Compute bias potential forces
        bias_forces = self.bias_potential.force_ana(x)

        return pair_forces + bias_forces
