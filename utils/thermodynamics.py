#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 15:28:49 2024

@author: bettina
"""
# -----------------------------------------
#   I M P O R T S
# -----------------------------------------
import numpy as np
import scipy.constants as const
from scipy.integrate import quad


class StatisticalThermodynamics:
    def __init__(self):
        """
        Initialize StatisticalThermodynamics class.

        Default integration limits x_low = -inf and x_high = +inf.
        """
        self.x_low, self.x_high = -np.inf, np.inf
        self.v_low, self.v_high = -np.inf, np.inf

    def boltzmann_factor(self, x, potential, T):
        """
        Calculate the unnormalized Boltzmann factor.

        The unnormalized Boltzmann factor is given by:
        p(x) = exp(- V(x) / (R * T))

        where R is given in kJ / (mol K).
        The potential energy V(x) is given from the potential class in kJ/mol.

        Parameters:
            - potential (object): An object representing the potential energy landscape of the system.
                                  It should have a 'potential' method that calculates the potential at a given position in kJ/mol.
            - x (float): Position.
            - T (float): Temperature in units of K.

        Returns:
            float: The value of the unnormalized Boltzmann factor at the given position x.
        """
        R = const.R * 0.001  # Gas constant: conversion from J / (mol K) in kJ / (mol K)
        return np.exp(-potential.potential(x) / (T * R))

    # velocity density for one-dimensional systems
    def velocity_density_D1(self, v, m, T):
        """
        Calculate the one-dimensional velocity density as

        The unnormalized Boltzmann factor is given by:
        p(v) = sqrt(m /(2 * pi * R * T) ) * exp(- v * v * m  / (2 * R * T))


        Parameters:
            - v (float): velocity
            - m (float): mass
            - T (float): temperature in units of K

        Returns:
            float: The value of the one-dimensional velocity densityat the given velocity v.
        """
        # Get natural constants in the appropriate units
        R = const.R * 0.001

        return np.sqrt(m / (2 * const.pi * R * T)) * np.exp(-(v * v * m) / (2 * R * T))

    def pot_partition_function(self, potential, T, limits=None):
        r"""
        Calculate the partition function for the 1-dimensional potentials.

        The partition function is given by:
        Q = \int_{-\infty}^{\infty} p(x) dx

        In practice, the integration bounds are set to specific values, specified in the variable limits.
        Integration is carried out by scipy.integrate.quad.

        Parameters:
            - potential (object): An object representing the potential energy landscape of the system.
                                  It should have a 'potential' method that calculates the potential at a given position in kJ/mol.
            - T (float): Temperature in units of K.
            - limits (tuple, list, optional): Integration limits in nm.
                                              Defaults of x_low, x_high = -inf, +inf have been set at the initialization of the class.

        Returns:
            ndarray: The value of the partition function and an error estimate array([Q, Q_error]) in units of nm.
        """
        # print(f"Calculating the classical partition function for the {potential.__class__.__name__} potential.")
        # print("----------------------------------------------------------------------")

        if limits is None:
            x_low, x_high = self.x_low, self.x_high
            # print(f"No integration limits defined. Using default limits: ({self.x_low}, {self.x_high})")
        else:
            if isinstance(limits, (list, tuple)) and len(limits) == 2:
                x_low, x_high = limits
            else:
                raise ValueError(
                    "Input 'limits' is not a list or tuple with two elements."
                )

        try:
            Q, Q_error = quad(
                lambda x: self.boltzmann_factor(x, potential, T), x_low, x_high
            )
            return np.array([Q, Q_error])
        except Exception as e:
            print(f"Error: {e}")
            return False

    def kin_partition_function(self, m, T):
        r"""
        Calculate the partition function for the kinetic energy in one dimension.

        Q_kin       = \int_{-\infty}^{\infty} e^{-E_kin(v) / k_B * T} dv
        E_kin(v)    = (m / 2)  * (v)^2

        using gaussian integral: \int e^(-ax^2) dx = \sqrt{\pi / a}

        Q_kin = \sqrt{2 * pi * k_B * T / m}

        Scinse dv has units of v, Q_kin does aswell.

        Q_kin[nm/ps] = \sqrt{2 * pi * k_B[J/K] * T[K] / (m[u] * u)} * 1e-3[\frac{s}{m} \frac{nm}{ps}]


        Parameters:
            - m (float): Mass of the particle in u.
            - T (float): Temperature in units of K.

        Returns:
            float: The value of the kinetic energy partition function (kinetic phase integral) in units of ps/nm.
        """
        # Atomic mass unit in kg
        atomic_mass_unit = const.physical_constants["atomic mass constant"][0]

        m_kg = m * atomic_mass_unit

        k_B = const.Boltzmann  # Boltzmann constant in J/K
        Q_kin_SI = np.sqrt(2 * np.pi * k_B * T / m_kg)  # in m/s
        Q_kin = Q_kin_SI * 1e-3  # nm/ps

        return Q_kin

    def _boltzmann_factor_kin(self, v, m, T):
        r"""
        Calculate the kinetic Boltzmann factor function for the kinetic energy in one dimension.

        p(v, m, T) = e^{- ((m[u] * u[kg] / 2) * (v[nm/ps])^2) / (k_B[(nm^2 kg)/(ps^2 K)] * T[K])}

        with u being the atomic mass unit in kg, and k_B the Boltzmann constant in (m^2 kg)/(s^2 K) = J/K.


        Parameters:
            - v (float): Velocity in nm/ps.
            - m (float): Mass of the particle in u.
            - T (float): Temperature in units of K.


        Returns:
            float: The value of the kinetic energy Boltzmann factor.
        """

        # Atomic mass unit in kg (SI)
        atomic_mass_unit = const.physical_constants["atomic mass constant"][0]

        m_kg = m * atomic_mass_unit  # convert u to kg (SI)
        k_B = const.Boltzmann  # given in (m^2 kg)/(s^2 K) (SI)
        k_B = k_B * 1e-6  # given in (nm^2 kg)/(ps^2 K)
        return np.exp(
            -((m_kg / 2) * v**2) / (k_B * T)
        )  # returns unitless Boltzmann factor for kinetic energy

    def kin_partition_function_int(self, m, T, limits=None):
        r"""
        Calculate the partition function for the kinetic energy in one dimension.

        Q = \int_{-\infty}^{\infty} p(v) dv

        with p(v) being the kinetic energy Boltzmann factor


        Parameters:
            - m (float): Mass of the particle in u.
            - T (float): Temperature in units of K.
            - limits (tuple, list, optional): Integration limits in nm/ps.
                                              Defaults of v_low, v_high = -inf, +inf have been set at the initialization of the class.

        Returns:
            float: The value of the kinetic energy partition function in nm/ps.
        """

        if limits is None:
            v_low, v_high = self.v_low, self.v_high
            # print(f"No integration limits defined. Using default limits: ({self.v_low}, {self.v_high})")
        else:
            if isinstance(limits, (list, tuple)) and len(limits) == 2:
                v_low, v_high = limits
            else:
                raise ValueError(
                    "Input 'limits' is not a list or tuple with two elements."
                )

        try:
            Q_kin, Q_kin_error = quad(
                lambda v: self._boltzmann_factor_kin(v, m, T), v_low, v_high
            )
            return np.array([Q_kin, Q_kin_error])  # nm/ps
        except Exception as e:
            print(f"Error: {e}")
            return False


# #  Plot Boltzmann density for various temperatures
# if test_p == True:
#     print("---------------------------------")
#     print(" testing the Boltzmann density   ")
#     # set x-axis
#     x = np.linspace(0, 5, 501)

#     # set parameters of the potential
#     this_param = [2, 1, 5, 2, 1, 10]

#     # generate instance of the potential class
#     this_potential = D1.Bolhuis(this_param)

#     # plot the potential
#     plt.figure(figsize=(12, 6))
#     plt.plot(x, this_potential.potential(x), color='blue', label='V(x)')

#     plt.ylim(0,20)
#     plt.xlabel("x")
#     plt.ylabel("V(x)")
#     plt.title("Force for various values of alpha, line: analytical, dots: numerical")
#     plt.legend()

#     # check error handling in the function partition function
#     print("no limits passed:")
#     print("Q = ", this_potential.partition_function(300))
#     limits = 3.0
#     print("limits is a float:")
#     print("Q = ", this_potential.partition_function(300, limits))
#     limits = [1,2,3]
#     print("limits is a an array with wrong length:")
#     print("Q = ", this_potential.partition_function(300, limits))

#     # plot partition function as a function of temperature
#     T_min = 200
#     T_max = 500
#     T_list = np.linspace(T_min, T_max, 12)
#     Q_list = np.zeros( (12,2) )
#     N_list = np.zeros(12)

#     for i, T in enumerate(T_list):
#         Q_list[i]  =  this_potential.partition_function(T_list[i])

#     plt.figure(figsize=(12, 6))
#     plt.plot(T_list, Q_list[:,0])

#     plt.xlabel("T")
#     plt.ylabel("Q(T)")
#     plt.title("Partition functions as a function of temperature")

#     # plot Boltzmann densits for various temperatures
#     plt.figure(figsize=(12, 6))
#     for i, T in enumerate(T_list):
#         color = plt.cm.coolwarm((T-T_min) / (T_max-T_min) )  # Normalize a to be in [0, 1]

#         # calculate partition function
#         Q =  this_potential.partition_function(T)

#         # plot Boltzmann distrobution
#         plt.plot(x, this_potential.boltzmann_factor(x, T)/Q[0], color=color, label='T={:.0f}'.format(T))
#         plt.xlabel("x")
#         plt.ylabel("p(x)")
#         plt.title("Normalized Boltzmann density for various temperatures")


#     # check wethere thes Boltzmann densities are indeed normalized
#     # this is an indirect check whether Q is correct
#     # x grid needs to be large enough to cover most of p(x) left and right of the maximum
#     x_grid = np.linspace(-1, 5, 601)
#     # grid spacing
#     dx = 6 / 600

#     for i, T in enumerate(T_list):
#         # calcualate the normalization constant
#         # using the Riemann integral on the grid x_grid
#         N_list[i] = np.sum( this_potential.boltzmann_factor(x_grid, T) ) * (1 / Q_list[i,0])* dx

#     plt.figure(figsize=(12, 6))
#     plt.plot(T_list, N_list)

#     plt.xlabel("T")
#     plt.ylabel("N")
#     plt.title("Norm of the normalized Boltzmann density for various temperatures")
