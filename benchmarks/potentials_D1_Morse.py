#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed June 27 13:00:56 2024

@author: arthur

This tests the one-dimensional Morese potential 
implemented in the potential class

potential.D1(MorsePotential)

"""

#-----------------------------------------
#   I M P O R T S 
#-----------------------------------------

import sys
sys.path.append("..")  


import matplotlib.pyplot as plt
import numpy as np


#-----------------------------------------
#   D1_Morse
#-----------------------------------------
# local packages and modules
from potential import D1


# Example usage
D_e = 3.0  # Well depth
a = 1.0    # Width parameter
x_e = 1.0  # Equilibrium bond distance
param = [D_e, a, x_e]

morse = D1.MorsePotential(param)
x = 1.0

print("Potential at x =", x, ":", morse.potential(x))
print("Force at x =", x, ":", morse.force_ana(x))
print("Hessian at x =", x, ":", morse.hessian_ana(x))

h = 0.2

x_values = np.linspace(0.2, 6, 400)

# Calculate potential, force, and Hessian
V_values = morse.potential(x_values)
F_values = morse.force_ana(x_values).flatten()
H_values = morse.hessian_ana(x_values).flatten()

F_values_num = morse.force_num_s(x_values, h).flatten()
H_values_num = morse.hessian_num_s(x_values, h).flatten()

# Plot the potential, force, and Hessian
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(x_values, V_values, label="Morse Potential")
plt.xlabel("x")
plt.ylabel("V(x)")
plt.title("Morse Potential")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(x_values, F_values, label="Analytical Morse Force", color='red')
plt.plot(x_values, F_values_num, label="Finite Difference Morse Force", color='k', linestyle="--", alpha=1)
plt.xlabel("x")
plt.ylabel("F(x)")
plt.title("Force from Morse Potential")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(x_values, H_values, label="Analytical Morse Hessian", color='green')
plt.plot(x_values, H_values_num, label="Finite Difference Morse Hessian", color='k', linestyle="--", alpha=1)
plt.xlabel("x")
plt.ylabel("H(x)")
plt.title("Hessian from Morse Potential")
plt.legend()

plt.tight_layout()
plt.show()

plt.close()