#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 09:28:56 2024

@author: bettina

Next steps:
     
    
    #------------------------------------------
    # package "rate_theory" 
    
    - create a new package "rate_theory" 
    - within rate_theory, write a module D1 for rates from 1D-potentials
    - D1: TST
    - D1: Kramers
    - D1: SqRA
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

    - move Boltzmann factor and partition function to a different module

    - D1: DoubleWell
    - D1: GaussianBias
    - D1: TripleWell
    - D1: Prinz potential
    - D1: Linear potential 
    - D1: Harmonic potential
    - D1: Morse potential
    
    #------------------------------------------
    # documentation
    - within "manual" write the manual as md-files (theory, how to use the code)
    - add README files to each package that explain the code structure of the package
    
    #------------------------------------------
    # cook book
    - create a directory "cookbook" for example use cases.

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
#from rate_theory import D1


#-----------------------------------------
#   F U N C T I O N S
#-----------------------------------------

def TST_D1(x_A, x_TS, T, m, potential): 

    # get natural constants in the right units    
    h = const.h * const.Avogadro * 0.001 *  1e12
    R = const.R * 0.001
    
    # extract positions of A and TS, temperature and mass
    
    # get the energy barrier
    E_b = potential.potential(x_TS) - potential.potential(x_A)

    #get the force consant in the transition state
    k = potential.hessian(x_A)[0,0]
    
    # calculate the frequency
    nu = 1/(2* np.pi) * np.sqrt(k/m)
    
    #set vibrational partition function of the TS to one
    q_TS = 1
    
    # calculate the partition function of A
    q_A = np.exp(- h * nu / (2 * R * T)) /  (1 - np.exp(- h * nu / ( R * T)) )
    
    k_AB = R * T / h * (q_TS / q_A) * np.exp(- E_b / (R * T))    
    
    return k_AB


#-----------------------------------------
#   S Y S T E M
#-----------------------------------------
m = 100.0
x = 0.0
v = 0.0 
T = 300.0
xi = 1.0
dt = 0.001
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
param = [2, 2, 5 , 1, 2, 0]
potential = D1.Bolhuis(param)

# set x-axis
x = np.linspace(-2, 6, 501)

#set list of alpha values
alpha_list = np.linspace(0, 38, 20)

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
for i, alpha in enumerate(alpha_list):
    
    # change alpha in the class instance
    potential.alpha = alpha
    # plot
    color = plt.cm.viridis( alpha / len(alpha_list) )  # Normalize alpha to be in [0, 1]
    plt.plot(x, potential.potential(x) , color=color, label='alpha={:.2f}'.format(alpha))
    
plt.ylim(0,50)
plt.xlabel("x in nm")
plt.ylabel("V(x) in kJ/mol") 
plt.title("Vary parameter alpha")
plt.legend()

# reset alpha to zero
potential.alpha = 0


#-----------------------------------------
#   M I N I M A 
#-----------------------------------------

# initizalize lists for the two minima
min_1 = np.zeros( len(alpha_list) )
min_2 = np.zeros( len(alpha_list) )


# caclulate start values
min_1_start = potential.a - np.sqrt( potential.b )
min_2_start = potential.a + np.sqrt( potential.b )

# initizalize lists for the forces in the two minima
force_min_1 = np.zeros( len(alpha_list) )
force_min_2 = np.zeros( len(alpha_list) )

# initizalize lists for the hessians in the two minima
hessian_min_1 = np.zeros( len(alpha_list) )
hessian_min_2 = np.zeros( len(alpha_list) )


# loop over alpha
for i, alpha in enumerate(alpha_list):
    
    # change alpha in the class instance
    potential.alpha = alpha

    # find minima using scipy optimize
    min_1[i] = potential.min(min_1_start)
    min_2[i] = potential.min(min_2_start)

    # calculate the forces in the minima
    force_min_1[i] = potential.force( min_1[i] )        
    force_min_2[i] = potential.force( min_2[i] ) 
    
    # calculate the hessians in the minima
    hessian_min_1[i] = potential.hessian( min_1[i] )        
    hessian_min_2[i] = potential.hessian( min_2[i] ) 

# reset alpha to zero
potential.alpha = 0

# print position and force of the minima
print("-----------------------------------------------------------------------")
print(" Minimum 1 from alpha = ", alpha_list[0], " to  alpha = ", alpha_list[-1])
print("-----------------------------------------------------------------------")
print("Minimum 1: position")
print(min_1)
print("")
print("Minimum 1: force")
print(force_min_1)
print("")
print("Minimum 1: hessian")
print(hessian_min_1)
print("")
print("-----------------------------------------------------------------------")
print(" Minimum 2 from alpha = ", alpha_list[0], " to  alpha = ", alpha_list[-1])
print("-----------------------------------------------------------------------")
print("Minimum 2: position")
print(min_2)
print("")
print("Minimum 2: force")
print(force_min_2)
print("")
print("Minimum 2: hessian")
print(hessian_min_2)


#-----------------------------------------
#   T R A N S I T I O N   S T A T E
#-----------------------------------------

# initizalize list for the transition state
TS = np.zeros( len(alpha_list) )

# initizalize list for the force at the transition state
force_TS = np.zeros( len(alpha_list) )

# initizalize list for the hessian at the transition state
hessian_TS = np.zeros( len(alpha_list) )

# loop over alpha
for i, alpha in enumerate(alpha_list):
    
    # change alpha in the class instance
    potential.alpha = alpha
    
    # calculate transition state
    TS[i] = potential.TS( min_1[i], min_2[i])
    
    # calculate force at the transition state
    force_TS[i] = potential.force( TS[i] )

    # calculate hessian at the transition state
    hessian_TS[i] = potential.hessian( TS[i] )

# reset alpha to zero
potential.alpha = 0    
   
# print position and force of the transition state
print("")
print("-----------------------------------------------------------------------")
print(" Transition state from alpha = ", alpha_list[0], " to  alpha = ", alpha_list[-1])
print("-----------------------------------------------------------------------")
print("TS: position")
print(TS)
print("")
print("TS: force")
print(force_TS)
print("")
print("TS: hessian")
print(hessian_TS)


#-----------------------------------------
#   T S T  R A T E
#-----------------------------------------
print("")
print("-----------------------------------------------------------------------")
print(" TST rate")
print("-----------------------------------------------------------------------")


# initizalize list TST rates
k_AB = np.zeros( len(alpha_list) )
k_BA = np.zeros( len(alpha_list) )

# loop over alpha
for i, alpha in enumerate(alpha_list):

    # set alpha value in the potential
    potential.alpha = alpha_list[i]
    
    
    # calculate the TST rates
    k_AB[i] = TST_D1(min_1[i], TS[i], system.T, system.m, potential)
    k_BA[i] = TST_D1(min_2[i], TS[i], system.T, system.m, potential)

print("k_AB")
print(k_AB)
print("k_BA")
print(k_BA)


# vary parameter alpha
plt.figure(figsize=(12, 6)) 
    
plt.semilogy(alpha_list, k_AB, color="red", label='k_AB, TST')
plt.semilogy(alpha_list, k_BA, color="blue", label='k_BA, TST')

    
plt.xlabel("alpha")
plt.ylabel("rate constant in 1/ps") 
plt.title("rate constant")
plt.legend()