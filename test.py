#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 09:28:56 2024

@author: bettina

Next steps:     
    #------------------------------------------
    # package "potentials" 
    
    - D1_Bolhuis: implement function that returns the extrema
    - D1_Bolhuis: change the return type of the force function to numpy array
    
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


"""

#-----------------------------------------
#   I M P O R T S 
#-----------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const


#-----------------------------------------
#   D1_Bolhuis
#-----------------------------------------

# local packages and modules
from potentials import D1_Bolhuis

test_V = False
test_F = True
test_H = True
test_p = False



#  Plot potential for various parameters
if test_V == True: 
    print("---------------------------------")
    print(" testing the Potential   ")
    
    # basis test whether the function works as expected
    my_param = [2, 1, 20, 1, 0, 0]
    print(D1_Bolhuis.V(4, *my_param))



    # set x-axis
    x = np.linspace(-5, 5, 501)
    
    #----------------------
    # vary parameter a
    plt.figure(figsize=(12, 6)) 
    for a in np.linspace(-2,2, 11):     
        my_param = [a, 1, 20, 1, 0, 0]
        color = plt.cm.viridis((a + 2) / 4)  # Normalize a to be in [0, 1]
        plt.plot(x, D1_Bolhuis.V(x, *my_param), color=color, label='a={:.2f}'.format(a))
        
    plt.ylim(0,20)
    plt.xlabel("x")
    plt.ylabel("V(x)") 
    plt.title("Vary parameter a")
    plt.legend()
    
    #----------------------
    # vary parameter b
    plt.figure(figsize=(12, 6)) 
    for b in np.linspace(0, 4, 11):     
        my_param = [2, b, 20, 1, 0, 0]
        color = plt.cm.viridis(b / 4)  # Normalize a to be in [0, 1]
        plt.plot(x, D1_Bolhuis.V(x, *my_param), color=color, label='b={:.2f}'.format(b))
    
    plt.ylim(0,20)
    plt.xlabel("x")
    plt.ylabel("V(x)") 
    plt.title("Vary parameter b")
    plt.legend()
    
    #----------------------
    # vary parameter k1
    plt.figure(figsize=(12, 6)) 
    for k1 in np.linspace(0, 4, 11):     
        my_param = [2, 1, 20, k1, 0, 0]
        color = plt.cm.viridis(k1 / 4)  # Normalize a to be in [0, 1]
        plt.plot(x, D1_Bolhuis.V(x, *my_param), color=color, label='k1={:.2f}'.format(k1))
    
    plt.ylim(0,20)
    plt.xlabel("x")
    plt.ylabel("V(x)") 
    plt.title("Vary parameter k1")
    plt.legend()
    
    #----------------------
    # vary parameter k2
    plt.figure(figsize=(12, 6)) 
    for k2 in np.linspace(0, 4, 11):     
        my_param = [2, 1, 20, 2, k2, 0]
        color = plt.cm.viridis(k2 / 4)  # Normalize a to be in [0, 1]
        plt.plot(x, D1_Bolhuis.V(x, *my_param), color=color, label='k2={:.2f}'.format(k2))
    
    plt.ylim(0,20)
    plt.xlabel("x")
    plt.ylabel("V(x)") 
    plt.title("Vary parameter k2")
    plt.legend()
    
    
    #----------------------
    # vary parameter c
    plt.figure(figsize=(12, 6)) 
    for c in np.linspace(0, 40, 11):     
        my_param = [2, 1, c, 2, 0, 2]
        color = plt.cm.viridis(c / 40)  # Normalize a to be in [0, 1]
        plt.plot(x, D1_Bolhuis.V(x, *my_param), color=color, label='c={:.2f}'.format(c))
    
    plt.ylim(0,20)
    plt.xlabel("x")
    plt.ylabel("V(x)") 
    plt.title("Vary parameter c")
    plt.legend()
    
    #----------------------
    # vary parameter alpha
    plt.figure(figsize=(12, 6)) 
    for alpha in np.linspace(0, 4, 11):     
        my_param = [2, 1, 20, 2, 0, alpha]
        color = plt.cm.viridis(alpha / 4)  # Normalize a to be in [0, 1]
        plt.plot(x, D1_Bolhuis.V(x, *my_param), color=color, label='alpha={:.2f}'.format(alpha))
        
    plt.ylim(0,20)
    plt.xlabel("x")
    plt.ylabel("V(x)") 
    plt.title("Vary parameter alpha")
    plt.legend()

#  Compare force calculated analytically and numerically
if test_F == True: 
    print("---------------------------------")
    print(" testing the force   ")
    
    # set x-axis
    x = np.linspace(0, 5, 501)
    h = 0.001

    #----------------------
    # vary parameter alpha
    plt.figure(figsize=(6, 6)) 
    plt.axhline(y=0, color='black', linestyle='dotted', linewidth=1)
    
    for alpha in np.linspace(0, 4, 5):     
        my_param = [2, 1, 20, 2, 1, alpha]
        color = plt.cm.viridis(alpha / 4)  # Normalize a to be in [0, 1]
        
        # plot analytical force 
        plt.plot(x, D1_Bolhuis.F(x, *my_param), color=color, label='alpha={:.2f}'.format(alpha))
        # plot numericaal  force 
        plt.plot(x, D1_Bolhuis.F_num(x, h, *my_param), color=color, marker='o', linestyle='None', markersize=3)
    
    plt.ylim(-20,20)
    plt.xlabel("x")
    plt.ylabel("F(x)") 
    plt.title("Force for various values of alpha, line: analytical, dots: numerical")
    plt.legend()
    
    #----------------------
    # vary parameter alpha, plot differnece between analytical and numerical Hessian
    plt.figure(figsize=(6, 6)) 
    plt.axhline(y=0, color='black', linestyle='dotted', linewidth=1)
    
    for alpha in np.linspace(0, 4, 5):     
        my_param = [2, 1, 20, 2, 0, alpha]
        color = plt.cm.viridis(alpha / 4)  # Normalize a to be in [0, 1]
        
        # plot deviation between analytical and numerical Hessian 
        plt.plot(x, D1_Bolhuis.F(x, *my_param) - D1_Bolhuis.F_num(x, h, *my_param), color=color, label='alpha={:.2f}'.format(alpha))
    
    plt.ylim(-5,5)
    plt.xlabel("x")
    plt.ylabel("H(x)- H_num(x)") 
    plt.title("Deviation between analytical force and numerical force for various values of alpha")
    plt.legend()   
    
#  Compare Hessian calculated analytically and numerically
if test_H == True: 
    print("---------------------------------")
    print(" testing the Hessian   ")

    # set x-axis
    x = np.linspace(0, 5, 501)
    h = 0.002

    #----------------------
    # vary parameter alpha, plot analytical and numerical Hessian
    plt.figure(figsize=(6, 6)) 
    plt.axhline(y=0, color='black', linestyle='dotted', linewidth=1)
    
    for alpha in np.linspace(0, 4, 5):     
        my_param = [2, 1, 20, 2, 0, alpha]
        color = plt.cm.viridis(alpha / 4)  # Normalize a to be in [0, 1]
        
        # plot analytical Hessian 
        plt.plot(x, D1_Bolhuis.H(x, *my_param)[0,0,:], color=color, label='alpha={:.2f}'.format(alpha))
        # plot numericaal  Hessian 
        plt.plot(x, D1_Bolhuis.H_num(x, h, *my_param)[0,0,:], color=color, marker='o', linestyle='None', markersize=3)
    
    plt.ylim(-200,100)
    plt.xlabel("x")
    plt.ylabel("H(x)") 
    plt.title("Hessian for various values of alpha, line: analytical, dots: numerical")
    plt.legend()


    #----------------------
    # vary parameter alpha, plot differnece between analytical and numerical Hessian
    plt.figure(figsize=(6, 6)) 
    plt.axhline(y=0, color='black', linestyle='dotted', linewidth=1)
    
    for alpha in np.linspace(0, 4, 5):     
        my_param = [2, 1, 20, 2, 0, alpha]
        color = plt.cm.viridis(alpha / 4)  # Normalize a to be in [0, 1]
        
        # plot deviation between analytical and numerical Hessian 
        plt.plot(x, D1_Bolhuis.H(x, *my_param)[0,0,:] - D1_Bolhuis.H_num(x, h, *my_param)[0,0,:], color=color, label='alpha={:.2f}'.format(alpha))
    
    plt.ylim(-5,5)
    plt.xlabel("x")
    plt.ylabel("H(x)- H_num(x)") 
    plt.title("Deviation between analytical Hessian and numerical Hessian for various values of alpha")
    plt.legend()    
    
#  Plot Boltzmann density for various temperatures
if test_p == True: 
    print("---------------------------------")
    print(" testing the Boltzmann density   ")
    # set x-axis
    x = np.linspace(0, 5, 501)

    # set parameters of the potential
    my_param = [2, 1, 5, 2, 1, 10]
    
    # plot the potential
    plt.figure(figsize=(12, 6)) 
    plt.plot(x, D1_Bolhuis.V(x, *my_param), color='blue', label='V(x)')

    plt.ylim(0,20)
    plt.xlabel("x")
    plt.ylabel("V(x)") 
    plt.title("Force for various values of alpha, line: analytical, dots: numerical")
    plt.legend() 


    # plot partition function as a function of temperature
    T_min = 200
    T_max = 500 
    T_list = np.linspace(T_min, T_max, 12)
    Q_list = np.zeros( (12,2) )
    N_list = np.zeros(12)

    for i, T in enumerate(T_list): 
        Q_list[i]  = D1_Bolhuis.Q(T_list[i], *my_param)

    plt.figure(figsize=(12, 6)) 
    plt.plot(T_list, Q_list[:,0])
    
    plt.xlabel("T")
    plt.ylabel("Q(T)") 
    plt.title("Partition functions as a function of temperature")
    plt.legend()
        
    
    # plot Boltzmann densits for various temperatures
    plt.figure(figsize=(12, 6)) 
    for i, T in enumerate(T_list): 
        color = plt.cm.coolwarm((T-T_min) / (T_max-T_min) )  # Normalize a to be in [0, 1]
    
        # calculate partition function 
        Q =  D1_Bolhuis.Q(T, *my_param)
    
        # plot Boltzmann distrobution  
        plt.plot(x, D1_Bolhuis.p(x, T, *my_param)/Q[0], color=color, label='T={:.0f}'.format(T))
        plt.xlabel("x")
        plt.ylabel("p(x)") 
        plt.title("Normalized Boltzmann density for various temperatures")
        
   
    # check wethere thes Boltzmann densities are indeed normalized
    # this is an indirect check whether Q is correct
    # x grid needs to be large enough to cover most of p(x) left and right of the maximum
    x_grid = np.linspace(-1, 5, 601)
    # grid spacing 
    dx = 6 / 600

    for i, T in enumerate(T_list): 
        # calcualate the normalization constant 
        # using the Riemann integral on the grid x_grid
        N_list[i] = np.sum( D1_Bolhuis.p(x_grid, T, *my_param) ) * (1 / Q_list[i,0])* dx
 
    plt.figure(figsize=(12, 6)) 
    plt.plot(T_list, N_list)
    
    plt.xlabel("T")
    plt.ylabel("N") 
    plt.title("Norm of the normalized Boltzmann density for various temperatures")
    
    
print("---------------------------------")  
