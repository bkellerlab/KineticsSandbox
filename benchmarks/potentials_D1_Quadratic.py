#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 10:22:46 2024

This tests the one-dimensional quadratic potential 
implemented in the potential class

potential.D1(Quadratic)

@author: bettina
"""

import os

# -----------------------------------------
#   I M P O R T S
# -----------------------------------------
import sys

import matplotlib.pyplot as plt
import numpy as np

# local packages and modules
from potential import D1

# Get the absolute path of the parent directory
parent_dir = os.path.abspath("..")

# Append the parent directory to sys.path if it is not already included
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

print("---------------------------------------")
print("System path:")
print(sys.path)

# -----------------------------------------
#   D1
# -----------------------------------------

test_V = True
test_F = True
test_H = True
# test_TS = True


print("---------------------------------------")
print(" testing the class initialization   ")

my_param = [5, 2]
my_potential = D1.Quadratic(my_param)

print(" ")
print("class members: ")
print(dir(my_potential))
print(" ")
print("values of the parameters")
print("k: ", my_potential.k)
print("a: ", my_potential.a)
print(" ")
x = np.array([1, 2, 3, 4])

print("potential at x = ", x)
print(my_potential.potential(x))


# --------------------------------------------------------
#  Plot potential for various parameter values
if test_V:
    print("---------------------------------")
    print(" testing the potential   ")

    # set x-axis
    x = np.linspace(-5, 5, 501)

    # ----------------------
    # vary parameter k
    plt.figure(figsize=(12, 6))
    for k in np.linspace(-5, 5, 11):
        # set parameters
        this_param = [k, 0]
        # generate instance of the potential class
        this_potential = D1.Quadratic(this_param)

        # plot
        color = plt.cm.viridis((k + 5) / 10)  # Normalize a to be in [0, 1]
        plt.plot(
            x, this_potential.potential(x), color=color, label="k={:.2f}".format(k)
        )

    plt.grid()
    plt.ylim(-10, 10)
    plt.xlabel("x")
    plt.ylabel("V(x)")
    plt.title("Vary parameter k")
    plt.legend()

    # ----------------------
    # vary parameter a
    plt.figure(figsize=(12, 6))
    for a in np.linspace(-5, 5, 11):
        # set parameters
        this_param = [3, a]
        # generate instance of the potential class
        this_potential = D1.Quadratic(this_param)

        # plot
        color = plt.cm.viridis((a + 5) / 10)  # Normalize a to be in [0, 1]
        plt.plot(
            x, this_potential.potential(x), color=color, label="a={:.2f}".format(a)
        )

    plt.grid()
    plt.ylim(-10, 10)
    plt.xlabel("x")
    plt.ylabel("V(x)")
    plt.title("Vary parameter a")
    plt.legend()

# --------------------------------------------------------
#  Plot force for various parameter values
if test_F:
    print("---------------------------------")
    print(" testing the force   ")

    # set x-axis
    x = np.linspace(-5, 5, 501)
    h = 0.001

    # ----------------------
    # vary parameter k
    plt.figure(figsize=(12, 6))
    for k in np.linspace(-5, 5, 11):
        # set parameters
        this_param = [k, 0]
        # generate instance of the potential class
        this_potential = D1.Quadratic(this_param)

        # plot
        color = plt.cm.viridis((k + 5) / 10)  # Normalize a to be in [0, 1]

        # plot analytical force
        plt.plot(
            x,
            this_potential.force_ana(x),
            color=color,
            label="k={:.2f}".format(k),
        )
        # plot numericaal  force
        plt.plot(
            x,
            this_potential.force_num(x, h),
            color=color,
            marker="o",
            linestyle="None",
            markersize=3,
        )

    plt.grid()
    plt.ylim(-10, 10)
    plt.xlabel("x")
    plt.ylabel("F(x)")
    plt.title("Vary parameter k, a=0")
    plt.legend()

    # ----------------------
    # vary parameter a
    plt.figure(figsize=(12, 6))
    for a in np.linspace(-5, 5, 11):
        # set parameters
        this_param = [5, a]
        # generate instance of the potential class
        this_potential = D1.Quadratic(this_param)

        # plot
        color = plt.cm.viridis((a + 5) / 10)  # Normalize a to be in [0, 1]

        # plot analytical force
        plt.plot(
            x,
            this_potential.force_ana(x),
            color=color,
            label="a={:.2f}".format(a),
        )
        # plot numericaal  force
        plt.plot(
            x,
            this_potential.force_num(x, h),
            color=color,
            marker="o",
            linestyle="None",
            markersize=3,
        )

    plt.grid()
    plt.ylim(-10, 10)
    plt.xlabel("x")
    plt.ylabel("F(x)")
    plt.title("Vary parameter a, k = 5")
    plt.legend()


# --------------------------------------------------------
#  Plot Hessian

if test_H:
    print("---------------------------------")
    print(" testing the Hessian   ")

    # set x-axis
    x = np.linspace(-5, 5, 501)
    h = 0.001

    plt.figure(figsize=(12, 6))
    # set parameters
    this_param = [5, 2]
    # generate instance of the potential class
    this_potential = D1.Quadratic(this_param)

    # plot
    color = plt.cm.viridis(0)  # Normalize a to be in [0, 1]

    # plot analytical Hessian
    plt.plot(x, this_potential.hessian_ana(x), color=color, label="k=5, a=2")
    # plot numericaal  Hessian
    plt.plot(
        x,
        this_potential.hessian_num(x, h),
        color=color,
        marker="o",
        linestyle="None",
        markersize=3,
    )

    plt.grid()
    plt.ylim(-10, 10)
    plt.xlabel("x")
    plt.ylabel("H(x), k=5, a=0")
    plt.legend()

plt.show()