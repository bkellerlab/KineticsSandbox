#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 09:28:56 2024

@author: bettina

Next steps:
     
    
    #------------------------------------------
    # module "rate_theory" 
    
    - rate_theory: Kramers
    - rate_theory:: SqRA
    - within rate_theory, write a module rate_matrix for rates from rate_matrices
    - rateMatrix: SqRA_rate (via Berezhkovski, Szabo)    
    - rateMatrix: SqRA_rate_via_its (as inverse of ITS)    
    - rateMatrix: MSM_rate_via_its (as inverse of ITS)   

    #------------------------------------------
    # create a new package "integrators" 

    #------------------------------------------
    # create a new package "MSM" 

    #------------------------------------------
    # package "system" 
    
    - make v optional at initialization, draw v from Maxwell-Boltzmann dist

    #------------------------------------------
    # package "potentials" 

    - D1: DoubleWell
    - D1: GaussianBias
    - D1: TripleWell
    - D1: Prinz potential
    - D1: Linear potential 
    - D1: Harmonic potential
    - D1: Morse potential
    - D1: Lennard Jones potential
    
    #---44---------------------------------------
    # module "thermodynamics"
    
    - implement Boltzmann factor
    - implement partition function
    - implement entropy
    
    #------------------------------------------
    # documentation
    - add README files to each package that explain the code structure of the package
    
    
    #------------------------------------------
    # cook book

"""

#-----------------------------------------
#   I M P O R T S 
#-----------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const

# local packages and modules
from system import system
from potential import D1
from integrator import stochastic as sd
#from utils import rate_theory



#-----------------------------------------
#   S Y S T E M
#-----------------------------------------
m = 100.0
x = 0.0
v = 0.0 
T = 300.0
xi = 1.0
dt = 0.1
# initialize the system
system = system.D1(m, x, v, T, xi, dt)


print("-----------------------------------------------------------------------")
print(" Initialized system ")
print("-----------------------------------------------------------------------")
print("Mass: ", system.m, " u")
print("Position: ", system.x, " nm")
print("Velocity: ", system.v, " nm/ps")
print("Temperature: ", system.T, " K")
print("Collision frequency: ", system.xi, " 1/ps")
print("Time step: ", system.dt, " ps")


#-----------------------------------------
#   P O T E N T I A L 
#-----------------------------------------

# initialize one-dimensional Bolhuis potential
param = [2, 2, 5 , 1, 1, 4]
potential = D1.Bolhuis(param)

# set x-axis
x = np.linspace(-2, 6, 501)



print("-----------------------------------------------------------------------")
print(" Initialized 1-dimensional Bolhuis potential ")
print("-----------------------------------------------------------------------")
print("a: ", potential.a)
print("b: ", potential.b)
print("c: ", potential.c)
print("k1: ", potential.k1)
print("k2: ", potential.k2)
print("alpha: ", potential.alpha)


#----------------------
# vary parameter alpha
plt.figure(figsize=(12, 6)) 

plt.plot(x, potential.potential(x) , color="darkblue")
    
plt.ylim(0,50)
plt.xlim(-2,10)
plt.xlabel("x in nm")
plt.ylabel("V(x) in kJ/mol") 
plt.title("The potential energy function")
plt.legend()


#-----------------------------------------
#   S I M U L A T I O N 
#-----------------------------------------
#np.random.seed(42)

n_steps = 1_000_000
pos = np.zeros(n_steps)
vel = np.zeros(n_steps)

for k in range(n_steps):
    # perform an integration step
    sd.ABO(system, potential)
    # save position and velocity to an array
    pos[k] = system.x
    vel[k] = system.v    

print("-----------------------------------------------------------------------")
#print(eta)

plt.figure(figsize=(12, 6)) 
plt.plot(pos)

plt.figure(figsize=(12, 6)) 
counts, bins = np.histogram(pos, bins=100)
plt.stairs(counts, bins)

#plt.vlines(eta_mean, 0, 0.05)
#plt.vlines(eta_mean + eta_var, 0, 0.05, color="gray")
#plt.vlines(eta_mean - eta_var, 0, 0.05, color="gray")

print("-----------------------------------------------------------------------")
print("END ")
print("-----------------------------------------------------------------------")
