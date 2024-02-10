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
#from utils import rate_theory



#-----------------------------------------
#  F U N C T I O N S
#-----------------------------------------
# A step
def A_step(system, half_step ='False'):
    """
    Perform the A-step in a Langevin splitting integrator

    Parameters:
        - system (object): An object representing the physical system undergoing Langevin integration.
                  It should have attributes 'x' (position), 'v' (velocity), and 'dt' (time step).
        - half_step (bool, optional): If True, perform a half-step integration. Default is False, performing
                              a full-step integration. A half-step is often used in the velocity
                              Verlet algorithm for symplectic integration.

    Returns:
        None: The function modifies the 'x' (position) attribute of the provided system object in place.
    """
 
    # set time step, depending on whether a half- or full step is performed
    if half_step == True:
        dt = 0.5 * system.dt
    else:
        dt = system.dt
        
    system.x = system.x + system.v * dt
    
    return None

# B step
def B_step(system, potential, half_step ='False'):
    """
    Perform a Langevin integration B-step for a given system.

    Parameters:
    - system (object): An object representing the physical system undergoing Langevin integration.
                      It should have attributes 'v' (velocity), 'm' (mass), and 'x' (position).
    - potential (object): An object representing the potential energy landscape of the system.
                         It should have a 'force' method that calculates the force at a given position.
    - half_step (bool, optional): If True, perform a half-step integration. Default is False, performing
                                  a full-step integration. A half-step is often used in the velocity
                                  Verlet algorithm for symplectic integration.

    Returns:
    None: The function modifies the 'v' (velocity) attribute of the provided system object in place based on the
          force calculated by the provided potential object.
    """
    
    # set time step, depending on whether a half- or full step is performed
    if half_step == True:
        dt = 0.5 * system.dt
    else:
        dt = system.dt
    
    system.v = system.v + (1 / system.m) * dt * potential.force_num( system.x, 0.001 )[0]
    
    return None

# O_step
def O_step(system, half_step ='False', eta_k = 'None'):
    
    """
    Perform the O-step in a Langevin integrator.

     Parameters:
         - system (object): An object representing the physical system undergoing Langevin integration.
                   It should have attributes 'v' (velocity), 'm' (mass), 'xi' (friction coefficient),
                   'T' (temperature), 'dt' (time step).
         - half_step (bool, optional): If True, perform a half-step integration. Default is False, performing
                   a full-step integration. 
         - eta_k (float or None, optional): If provided, use the given value as the random noise term (eta_k)
                   in the Langevin integrator. If None, draw a new value from a Gaussian normal distribution.

     Returns:
     None: The function modifies the 'v' (velocity) attribute of the provided system object in place.
     """

    # get natural constants in the appropriate units    
    R = const.R * 0.001
    
    # set time step, depending on whether a half- or full step is performed
    if half_step == True:
        dt = 0.5 * system.dt
    else:
        dt = system.dt

    # if eta_k is not provided, draw eta_k from Gaussian normal distribution
    if eta_k == 'None':
        eta_k = np.random.normal()

    d = np.exp(- system.xi * dt)
    f_v = np.sqrt( R * system.T *  (1 / system.m)  * (1 - np.exp(-2 * system.xi * dt)) )

    system.v = d * system.v +  f_v * eta_k
    return None

# ABO integrator
def ABO(system, potential, eta_k = 'None'):
    """
    Perform a full Langevin integration step consisting of A-step, B-step, and O-step.

    Parameters:
    - system (object): An object representing the physical system undergoing Langevin integration.
                      It should have attributes 'x' (position), 'v' (velocity), 'm' (mass), 'xi'
                      (friction coefficient), 'T' (temperature), and 'dt' (time step).
    - potential (object): An object representing the potential energy landscape of the system.
                         It should have a 'force' method that calculates the force at a given position.
    - eta_k (float or None, optional): If provided, use the given value as the random noise term (eta_k)
                        in the Langevin integrator. If None, draw a new value from a Gaussian normal distribution.

    Returns:
    None: The function modifies the 'x' and 'v' attributes of the provided system object in place.
    """    
    
    A_step(system)
    B_step(system, potential)
    O_step(system, eta_k)     
    return None   
    

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
    ABO(system, potential)
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

# #-----------------------------------------
# #   M I N I M A 
# #-----------------------------------------a

# # initizalize lists for the two minima
# min_1 = np.zeros( len(alpha_list) )
# min_2 = np.zeros( len(alpha_list) )


# # caclulate start values
# min_1_start = potential.a - np.sqrt( potential.b )
# min_2_start = potential.a + np.sqrt( potential.b )

# # initizalize lists for the forces in the two minima
# force_min_1 = np.zeros( len(alpha_list) )
# force_min_2 = np.zeros( len(alpha_list) )

# # initizalize lists for the hessians in the two minima
# hessian_min_1 = np.zeros( len(alpha_list) )
# hessian_min_2 = np.zeros( len(alpha_list) )


# # loop over alpha
# for i, alpha in enumerate(alpha_list):
    
#     # change alpha in the class instance
#     potential.alpha = alpha

#     # find minima using scipy optimize
#     min_1[i] = potential.min(min_1_start)
#     min_2[i] = potential.min(min_2_start)

#     # calculate the forces in the minima
#     force_min_1[i] = potential.force( min_1[i] )        
#     force_min_2[i] = potential.force( min_2[i] ) 
    
#     # calculate the hessians in the minima
#     hessian_min_1[i] = potential.hessian( min_1[i] )        
#     hessian_min_2[i] = potential.hessian( min_2[i] ) 

# # reset alpha to zero
# potential.alpha = 0

# # print position and force of the minima
# print("-----------------------------------------------------------------------")
# print(" Minimum 1 from alpha = ", alpha_list[0], " to  alpha = ", alpha_list[-1])
# print("-----------------------------------------------------------------------")
# print("Minimum 1: position")
# print(min_1)
# print("")
# print("Minimum 1: force")
# print(force_min_1)
# print("")
# print("Minimum 1: hessian")
# print(hessian_min_1)
# print("")
# print("-----------------------------------------------------------------------")
# print(" Minimum 2 from alpha = ", alpha_list[0], " to  alpha = ", alpha_list[-1])
# print("-----------------------------------------------------------------------")
# print("Minimum 2: position")
# print(min_2)
# print("")
# print("Minimum 2: force")
# print(force_min_2)
# print("")
# print("Minimum 2: hessian")
# print(hessian_min_2)


# #-----------------------------------------
# #   T R A N S I T I O N   S T A T E
# #-----------------------------------------

# # initizalize list for the transition state
# TS = np.zeros( len(alpha_list) )

# # initizalize list for the force at the transition state
# force_TS = np.zeros( len(alpha_list) )

# # initizalize list for the hessian at the transition state
# hessian_TS = np.zeros( len(alpha_list) )

# # loop over alpha
# for i, alpha in enumerate(alpha_list):
    
#     # change alpha in the class instance
#     potential.alpha = alpha
    
#     # calculate transition state
#     TS[i] = potential.TS( min_1[i], min_2[i])
    
#     # calculate force at the transition state
#     force_TS[i] = potential.force( TS[i] )

#     # calculate hessian at the transition state
#     hessian_TS[i] = potential.hessian( TS[i] )

# # reset alpha to zero
# potential.alpha = 0    
   
# # print position and force of the transition state
# print("")
# print("-----------------------------------------------------------------------")
# print(" Transition state from alpha = ", alpha_list[0], " to  alpha = ", alpha_list[-1])
# print("-----------------------------------------------------------------------")
# print("TS: position")
# print(TS)
# print("")
# print("TS: force")
# print(force_TS)
# print("")
# print("TS: hessian")
# print(hessian_TS)





# #-----------------------------------------
# #   T S T  R A T E
# #-----------------------------------------
# print("")
# print("-----------------------------------------------------------------------")
# print(" TST rate")
# print("-----------------------------------------------------------------------")


# # initizalize list of TST rates
# k_AB = np.zeros( len(alpha_list) )
# k_BA = np.zeros( len(alpha_list) )
# k_AB_ht = np.zeros( len(alpha_list) )
# k_BA_ht = np.zeros( len(alpha_list) )

# # loop over alpha
# for i, alpha in enumerate(alpha_list):

#     # set alpha value in the potential
#     potential.alpha = alpha_list[i]
    
#     # calculate the Eyring TST rates
#     k_AB[i] = rate_theory.TST_D1(min_1[i], TS[i], system.T, system.m, potential)
#     k_BA[i] = rate_theory.TST_D1(min_2[i], TS[i], system.T, system.m, potential)

#     # calculate the high-temperature approximation of the Eyring TST rates
#     k_AB_ht[i] = rate_theory.TST_ht_D1(min_1[i], TS[i], system.T, system.m, potential)
#     k_BA_ht[i] = rate_theory.TST_ht_D1(min_2[i], TS[i], system.T, system.m, potential)

# print("k_AB")
# print(k_AB)
# print("k_BA")
# print(k_BA)
# print("k_AB, ht")
# print(k_AB_ht)
# print("k_BA, ht")
# print(k_BA_ht)

# # vary parameter alpha
# plt.figure(figsize=(12, 6)) 
    
# plt.semilogy(alpha_list, k_AB, color="red", label='k_AB, TST')
# plt.semilogy(alpha_list, k_BA, color="blue", label='k_BA, TST')
# plt.semilogy(alpha_list, k_AB_ht, color="darkred", marker='o', linestyle='dashed', label='k_AB, high temperature TST')
# plt.semilogy(alpha_list, k_BA_ht, color="darkblue", marker='o', linestyle='dashed', label='k_BA, high temperature TST')

    
# plt.xlabel("alpha")
# plt.ylabel("rate constant in 1/ps") 
# plt.title("rate constant")
# plt.legend()