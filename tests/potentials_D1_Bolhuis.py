#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 09:28:56 2024

@author: bettina

This tests the one-dimensional Bolhuis potential 
implemented in the potential class

potential.D1(Bolhuis)

"""

#-----------------------------------------
#   I M P O R T S 
#-----------------------------------------
import sys
sys.path.append("..")  

import matplotlib.pyplot as plt
import numpy as np


#-----------------------------------------
#   D1_Bolhuis
#-----------------------------------------
# local packages and modules
from potentials import D1

test_V = True
test_F = False
test_H = False
test_p = False
test_TS = True

#-----------------------------------------
#   D1_Bolhuis  - Class
#-----------------------------------------
#  Plot potential for various parameters
if test_V == True: 
    print("---------------------------------")
    print(" testing the Potential   ")
    
    
    my_param = [2, 1, 5 , 1, 2, 10]
    my_potential = D1.Bolhuis(my_param)
    
    print(" ")
    print("class members: ")
    print(dir(my_potential))
    print(" ")
    print("values of the parameters")
    print("a: ",  my_potential.a, ", ", 
          "b: ",  my_potential.b, ", ",  
          "c: ",  my_potential.c, ", ",  
          "k1: ",  my_potential.k1, ", ",  
          "k2: ",  my_potential.k2, ", ",  
          "alpha: ",  my_potential.alpha)
    print(" ")
    x = np.array([1, 3, 4])
    
    print("potential at x = ", x)
    print(my_potential.potential(x))    

    # set x-axis
    x = np.linspace(-5, 5, 501)
    
    #----------------------
    # vary parameter a
    plt.figure(figsize=(12, 6)) 
    for a in np.linspace(-2,2, 11):
        # set parameters
        this_param = [a, 1, 20, 1, 0, 0]
        # generate instance of the potential class
        this_potential = D1.Bolhuis(this_param)
        
        # plot
        color = plt.cm.viridis((a + 2) / 4)  # Normalize a to be in [0, 1]
        plt.plot(x, this_potential.potential(x) , color=color, label='a={:.2f}'.format(a))
        
    plt.ylim(0,20)
    plt.xlabel("x")
    plt.ylabel("V(x)") 
    plt.title("Vary parameter a")
    plt.legend()
    
    #----------------------
    # vary parameter b
    plt.figure(figsize=(12, 6)) 
    for b in np.linspace(0, 4, 11):  
        # set parameters
        this_param = [2, b, 20, 1, 0, 0]
        # generate instance of the potential class
        this_potential = D1.Bolhuis(this_param)
        #plot
        color = plt.cm.viridis(b / 4)  # Normalize a to be in [0, 1]
        plt.plot(x, this_potential.potential(x), color=color, label='b={:.2f}'.format(b))
    
    plt.ylim(0,20)
    plt.xlabel("x")
    plt.ylabel("V(x)") 
    plt.title("Vary parameter b")
    plt.legend()
    
 
    #----------------------
    # vary parameter k1
    plt.figure(figsize=(12, 6)) 
    for k1 in np.linspace(0, 4, 11):     
        # set parameters        
        this_param = [2, 1, 20, k1, 0, 0]
        # generate instance of the potential class
        this_potential = D1.Bolhuis(this_param)
        #plot
        color = plt.cm.viridis(k1 / 4)  # Normalize a to be in [0, 1]
        plt.plot(x, this_potential.potential(x), color=color, label='k1={:.2f}'.format(k1))
    
    plt.ylim(0,20)
    plt.xlabel("x")
    plt.ylabel("V(x)") 
    plt.title("Vary parameter k1")
    plt.legend()   
 
    #----------------------
    # vary parameter k2
    plt.figure(figsize=(12, 6)) 
    for k2 in np.linspace(0, 4, 11):     
        # set parameters           
        this_param = [2, 1, 20, 2, k2, 0]
        # generate instance of the potential class
        this_potential = D1.Bolhuis(this_param)
        # plot
        color = plt.cm.viridis(k2 / 4)  # Normalize a to be in [0, 1]
        plt.plot(x, this_potential.potential(x), color=color, label='k2={:.2f}'.format(k2))
    
    plt.ylim(0,20)
    plt.xlabel("x")
    plt.ylabel("V(x)") 
    plt.title("Vary parameter k2")
    plt.legend()   
 
    #----------------------
    # vary parameter c
    plt.figure(figsize=(12, 6)) 
    for c in np.linspace(0, 40, 11):     
        # set parameters          
        this_param = [0, 1, c, 2, 0, 2]
        # generate instance of the potential class    
        this_potential = D1.Bolhuis(this_param)
        # plot         
        color = plt.cm.viridis(c / 40)  # Normalize a to be in [0, 1]
        plt.plot(x, this_potential.potential(x), color=color, label='c={:.2f}'.format(c))
    
    plt.ylim(0,20)
    plt.xlabel("x")
    plt.ylabel("V(x)") 
    plt.title("Vary parameter c")
    plt.legend()
    
    #----------------------
    # vary parameter alpha
    plt.figure(figsize=(12, 6)) 
    for alpha in np.linspace(0, 4, 11):  
        # set parameters          
        this_param = [0, 1, 20, 2, 0, alpha]
        # generate instance of the potential class    
        this_potential = D1.Bolhuis(this_param)
        # plot         
        color = plt.cm.viridis(alpha / 4)  # Normalize a to be in [0, 1]
        plt.plot(x, this_potential.potential(x), color=color, label='alpha={:.2f}'.format(alpha))
        
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
        # set parameters          
        this_param = [2, 1, 20, 2, 1, alpha]
        # generate instance of the potential class
        this_potential = D1.Bolhuis(this_param)
        # colors
        color = plt.cm.viridis(alpha / 4)  # Normalize a to be in [0, 1]
        
        # plot analytical force 
        plt.plot(x, this_potential.force(x)[0,:], color=color, label='alpha={:.2f}'.format(alpha))
        # plot numericaal  force 
        plt.plot(x, this_potential.force_num(x, h)[0,:], color=color, marker='o', linestyle='None', markersize=3)
    
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
        # set parameters  
        this_param = [2, 1, 20, 2, 1, alpha]
        # generate instance of the potential class
        this_potential = D1.Bolhuis(this_param)
        # colors
        color = plt.cm.viridis(alpha / 4)  # Normalize a to be in [0, 1]
        
        # plot deviation between analytical and numerical Hessian 
        plt.plot(x, this_potential.force(x)[0,:] - this_potential.force_num(x, h)[0,:], color=color, label='alpha={:.2f}'.format(alpha))
    
    plt.ylim(-5,5)
    plt.xlabel("x")
    plt.ylabel("F(x)- F_num(x)") 
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
        # set parameters
        this_param = [2, 1, 20, 2, 0, alpha]
        # generate instance of the potential class        
        this_potential = D1.Bolhuis(this_param)
        # colors
        color = plt.cm.viridis(alpha / 4)  # Normalize a to be in [0, 1]
        
        # plot analytical Hessian 
        plt.plot(x,  this_potential.hessian(x)[0,0,:], color=color, label='alpha={:.2f}'.format(alpha))
        # plot numericaal  Hessian 
        plt.plot(x, this_potential.hessian_num(x, h)[0,0,:], color=color, marker='o', linestyle='None', markersize=3)
    
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
        # set parameters
        this_param = [2, 1, 20, 2, 0, alpha]
        # generate instance of the potential class        
        this_potential = D1.Bolhuis(this_param)
        # colors
        color = plt.cm.viridis(alpha / 4)  # Normalize a to be in [0, 1]
        
        # plot deviation between analytical and numerical Hessian 
        plt.plot(x, this_potential.hessian(x)[0,0,:] - this_potential.hessian_num(x, h)[0,0,:], color=color, label='alpha={:.2f}'.format(alpha))
    
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
    this_param = [2, 1, 5, 2, 1, 10]
    
    # generate instance of the potential class
    this_potential = D1.Bolhuis(this_param)
    
    # plot the potential
    plt.figure(figsize=(12, 6)) 
    plt.plot(x, this_potential.potential(x), color='blue', label='V(x)')

    plt.ylim(0,20)
    plt.xlabel("x")
    plt.ylabel("V(x)") 
    plt.title("Force for various values of alpha, line: analytical, dots: numerical")
    plt.legend() 

    # check error handling in the function partition function
    print("no limits passed:")
    print("Q = ", this_potential.partition_function(300))
    limits = 3.0
    print("limits is a float:")
    print("Q = ", this_potential.partition_function(300, limits))
    limits = [1,2,3]
    print("limits is a an array with wrong length:")    
    print("Q = ", this_potential.partition_function(300, limits))    

    # plot partition function as a function of temperature
    T_min = 200
    T_max = 500 
    T_list = np.linspace(T_min, T_max, 12)
    Q_list = np.zeros( (12,2) )
    N_list = np.zeros(12)

    for i, T in enumerate(T_list): 
        Q_list[i]  =  this_potential.partition_function(T_list[i])

    plt.figure(figsize=(12, 6)) 
    plt.plot(T_list, Q_list[:,0])
    
    plt.xlabel("T")
    plt.ylabel("Q(T)") 
    plt.title("Partition functions as a function of temperature")
    
    # plot Boltzmann densits for various temperatures
    plt.figure(figsize=(12, 6)) 
    for i, T in enumerate(T_list): 
        color = plt.cm.coolwarm((T-T_min) / (T_max-T_min) )  # Normalize a to be in [0, 1]
    
        # calculate partition function 
        Q =  this_potential.partition_function(T)
    
        # plot Boltzmann distrobution  
        plt.plot(x, this_potential.boltzmann_factor(x, T)/Q[0], color=color, label='T={:.0f}'.format(T))
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
        N_list[i] = np.sum( this_potential.boltzmann_factor(x_grid, T) ) * (1 / Q_list[i,0])* dx
 
    plt.figure(figsize=(12, 6)) 
    plt.plot(T_list, N_list)
    
    plt.xlabel("T")
    plt.ylabel("N") 
    plt.title("Norm of the normalized Boltzmann density for various temperatures")

# Test whether we can find minima and maxima of the potential
if test_TS == True: 
    print("---------------------------------")
    print(" testing the ransition state finder")
    
    # keep alpha=0, k2=0 to allow for an analytical solution
    my_param = [2, 6, 0, 2, 0, 0]
    my_potential = D1.Bolhuis(my_param)    
    min_1 = my_param[0] + np.sqrt(my_param[1])
    min_2 = my_param[0] - np.sqrt(my_param[1])
    TS = my_param[0]
    
    print (" ")
    print ("Analytical solution ")
    print ("transition state:", TS, "force: ",  my_potential.force(TS), "hessian: ", my_potential.hessian(TS)) 

    print (" ")
    print ("Numerical solution provided by class")
    x_start = 2.01
    TS_num = my_potential.TS(min_1, min_2)
    print ("extremum:", TS_num, "force: ",  my_potential.force(TS_num), "hessian: ", my_potential.hessian(TS_num)) 
    
    
    
print("---------------------------------")  
