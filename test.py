#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 09:28:56 2024

@author: bettina

Next steps:
    
    #------------------------------------------  
    # package "utils"
    #------------------------------------------
    # module "rate_theory" 

    - implement Kramers
    - implement transition count across barrier
    - implement Peter's rate estimate

    #------------------------------------------
    # module "thermodynamics"

    - implement Maxwell-Boltzmann factor    
    - implement partition function
    - implement entropy

    #------------------------------------------------
    # create a new module "FP_discretized" 
    
    - rate_theory:: SqRA
    - within rate_theory, write a module rate_matrix for rates from rate_matrices
    - rateMatrix: SqRA_rate (via Berezhkovski, Szabo)    
    - rateMatrix: SqRA_rate_via_its (as inverse of ITS)    
    - rateMatrix: MSM_rate_via_its (as inverse of ITS)   

    #------------------------------------------------
    # create a new module "Markov_model" 

    #------------------------------------------
    # package "integrators" 
    #------------------------------------------
    # module "stochastic"
    - implement EM algorithm
    - implement integrators for biased potentials

    #------------------------------------------
    # package "system" 
    
    - make v optional at initialization, draw v from Maxwell-Boltzmann dist
    - add variables for bias force and random number
    - add variables for force -> more efficient implementation of Langevin integrators

    #------------------------------------------
    # package "potentials" 
    #------------------------------------------
    - D1: DoubleWell
    - D1: GaussianBias
    - D1: TripleWell
    - D1: Prinz potential
    - D1: Linear potential 
    - D1: Harmonic potential
    - D1: Morse potential
    - D1: Lennard Jones potential
    
    #------------------------------------------
    # documentation
    #------------------------------------------
    - update main README file
    - add README files to each package that explain the code structure of the package
    
    #------------------------------------------
    # cook book
    #------------------------------------------
    - Langevin simulation

    #------------------------------------------
    # application
    #------------------------------------------
    - Simulate Bolhuis potential for various values of alpha, 5 traj each, Langevin dynamics
        - 1e5 steps
        - 1e6 steps
        - 1e7 steps
        - 1e8 steps (?)
    - Estimate rate via transition count
    - Estimate rate via Peter's estimate
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
from utils import thermodynamics as thermo



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

n_steps = 10_000_000
n_steps_out = 10

pos = np.array([system.x])
vel = np.array([system.x])

for k in range(n_steps):
    # perform an integration step
    sd.ABO(system, potential)
    # save position and velocity to an array
    if k % n_steps_out == 0:
        pos = np.append(pos, system.x)
        vel = np.append(vel, system.v)

print("-----------------------------------------------------------------------")
#print(eta)

plt.figure(figsize=(12, 6)) 
plt.plot(pos)

# get a histogram of the sampled density
counts, bins = np.histogram(pos, bins=100)
# normalize counts
counts = counts.astype(float) / np.sum(counts)

# get analytical boltzmann density
boltzmann_density = thermo.boltzmann_factor(potential, bins[0:-1], system.T)
# normalize boltzmann_density
boltzmann_density = boltzmann_density / np.sum(boltzmann_density)



plt.figure(figsize=(12, 6)) 
plt.stairs(counts, bins, label="sampled distribution")
plt.stairs(boltzmann_density, bins, label="Boltzmann distribution")
plt.legend(fontsize=16)

#plt.vlines(eta_mean, 0, 0.05)
#plt.vlines(eta_mean + eta_var, 0, 0.05, color="gray")
#plt.vlines(eta_mean - eta_var, 0, 0.05, color="gray")

print("-----------------------------------------------------------------------")
print("END ")
print("-----------------------------------------------------------------------")
