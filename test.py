#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 09:28:56 2024

@author: bettina

Next steps:     
    #------------------------------------------
    # package "potentials" 

    - D1_Bolhuis: test unnormalized Boltzmann dist
    - D1_Bolhuis: implement partition function
    - D1_Bolhuis: implement function that returns the extrema
    - D1_Bolhius: implement first derivative, analytically and numerically
    - D1_Bolhius: implement second derivative, analytically and numerically
    
    #------------------------------------------
    # package "rate_constant" 
    
    - create a new package "rate_constants" 
    - within rate_constants, write a module D1 for rates from 1D-potentials
    - D1: TST
    - D1: Kramers
    - D1: SqRA
    - D1: SqRA_rate (via Berezhkovski, Szabo)    
    - D1: SqRA_rate_via_its (as inverse of ITS)    
    - D1: MSM_rate_via_its (as inverse of ITS)   

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

test_V = True
test_F = True


# basis test whether the function works as expected
my_param = [2, 1, 20, 1, 0, 0]
print(D1_Bolhuis.V(4, *my_param))

#  Plot potential for various parameters
if test_V == True: 

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
    # set x-axis
    x = np.linspace(0, 5, 501)
    h = 0.02

    #----------------------
    # vary parameter alpha
    plt.figure(figsize=(6, 6)) 
    plt.axhline(y=0, color='black', linestyle='dotted', linewidth=1)
    
    for alpha in np.linspace(0, 4, 5):     
        my_param = [2, 1, 20, 2, 0, alpha]
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


