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
    # package "potentials" 

    - D1: DoubleWell
    - D1: GaussianBias
    - D1: TripleWell
    - D1: Prinz potential
    - D1: Linear potential 
    - D1: Harmonic potential
    - D1: Morse potential
    
    #------------------------------------------
    # documentation
    - create directory "manual"
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
from scipy.optimize import minimize

# local packages and modules
from potentials import D1


#-----------------------------------------
#   S Y S T E M
#-----------------------------------------
mass = 1

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
plt.xlabel("x")
plt.ylabel("V(x)") 
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
    min_1[i] = minimize( potential.potential, min_1_start, method='BFGS' ).x
    min_2[i] = minimize( potential.potential, min_2_start, method='BFGS' ).x

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

