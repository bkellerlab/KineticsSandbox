#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 2025

@author: Ahmet Sarigun

This module implements deterministic integrators for one-dimensional systems.
"""

#-----------------------------------------
#   I M P O R T S 
#-----------------------------------------
import numpy as np

def euler(system, potential, n_steps):
    """
    Perform Euler integration of Newton's equations of motion.
    
    Parameters:
    - system (object): An object representing the physical system.
                      It should have attributes 'x' (position), 'v' (velocity), 
                      'm' (mass), and 'dt' (time step).
    - potential (object): An object representing the potential energy landscape.
                         It should have a 'force' method that calculates the force 
                         at a given position.
    - n_steps (int): Number of integration steps to perform.
    
    Returns:
    - tuple: (positions, velocities) arrays containing the trajectories
    """
    # Initialize arrays to store trajectories
    positions = np.zeros(n_steps + 1)
    velocities = np.zeros(n_steps + 1)
    
    # Store initial conditions
    positions[0] = system.x
    velocities[0] = system.v
    
    # Integration loop
    for t in range(n_steps):
        # Calculate force
        force = potential.force(positions[t], system.h)[0]
        
        # Update position and velocity using Euler method
        positions[t + 1] = positions[t] + velocities[t] * system.dt + \
                          (force/(2 * system.m)) * system.dt * system.dt
        velocities[t + 1] = velocities[t] + (force/system.m) * system.dt
        
        # Update system state
        system.x = positions[t + 1]
        system.v = velocities[t + 1]
    
    return positions, velocities

def verlet(system, potential, n_steps):
    """
    Perform Verlet integration of Newton's equations of motion.
    
    Parameters:
    - system (object): An object representing the physical system.
                      It should have attributes 'x' (position), 'v' (velocity), 
                      'm' (mass), and 'dt' (time step).
    - potential (object): An object representing the potential energy landscape.
                         It should have a 'force' method that calculates the force 
                         at a given position.
    - n_steps (int): Number of integration steps to perform.
    
    Returns:
    - tuple: (positions, velocities) arrays containing the trajectories
    """
    # Initialize arrays to store trajectories
    positions = np.zeros(n_steps + 1)
    velocities = np.zeros(n_steps + 1)
    
    # Store initial conditions
    positions[0] = system.x
    velocities[0] = system.v
    
    # Calculate initial force
    force = potential.force(positions[0], system.h)[0]
    
    # First position using Euler step
    positions[1] = positions[0] + velocities[0] * system.dt + \
                  (force/(2 * system.m)) * system.dt * system.dt
    
    # Main integration loop
    for t in range(1, n_steps):
        # Calculate force at current position
        force = potential.force(positions[t], system.h)[0]
        
        # Update position using Verlet algorithm
        positions[t + 1] = 2 * positions[t] - positions[t - 1] + \
                          (force/system.m) * system.dt * system.dt
        
        # Calculate velocity (central difference)
        velocities[t] = (positions[t + 1] - positions[t - 1]) / (2 * system.dt)
        
        # Update system state
        system.x = positions[t + 1]
        system.v = velocities[t]
    
    # Calculate final velocity
    velocities[-1] = (positions[-1] - positions[-2]) / system.dt
    
    return positions, velocities

def leapfrog(system, potential, n_steps):
    """
    Perform Leap Frog integration of Newton's equations of motion.
    
    Parameters:
    - system (object): An object representing the physical system.
                      It should have attributes 'x' (position), 'v' (velocity), 
                      'm' (mass), and 'dt' (time step).
    - potential (object): An object representing the potential energy landscape.
                         It should have a 'force' method that calculates the force 
                         at a given position.
    - n_steps (int): Number of integration steps to perform.
    
    Returns:
    - tuple: (positions, velocities) arrays containing the trajectories
    """
    # Initialize arrays to store trajectories
    positions = np.zeros(n_steps + 1)
    velocities = np.zeros(n_steps + 1)
    velocity_halfsteps = np.zeros(n_steps + 2)  # +2 for initial and final half steps
    
    # Store initial conditions
    positions[0] = system.x
    velocities[0] = system.v
    
    # Calculate initial force
    force = potential.force(positions[0], system.h)[0]
    
    # Initialize velocity half steps
    velocity_halfsteps[0] = system.v - (force/system.m) * (system.dt/2)
    velocity_halfsteps[1] = system.v + (force/system.m) * (system.dt/2)
    
    # Main integration loop
    for t in range(n_steps):
        # Calculate force at current position
        force = potential.force(positions[t], system.h)[0]
        
        # Update velocity at half step
        velocity_halfsteps[t + 1] = velocity_halfsteps[t] + \
                                   (force/system.m) * system.dt
        
        # Update position using half-step velocity
        positions[t + 1] = positions[t] + velocity_halfsteps[t + 1] * system.dt
        
        # Calculate velocity at full step (for output)
        velocities[t + 1] = (velocity_halfsteps[t + 1] - 
                            velocity_halfsteps[t]) / system.dt
        
        # Update system state
        system.x = positions[t + 1]
        system.v = velocities[t + 1]
    
    return positions, velocities
