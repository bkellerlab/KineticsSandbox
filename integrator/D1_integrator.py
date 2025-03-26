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

#-----------------------------------------
#   S I N G L E - S T E P   I N T E G R A T O R S
#-----------------------------------------

def euler_step(system, potential):
    """
    Perform a single Euler integration step for Newton's equations of motion.
    
    Parameters:
    - system (object): An object representing the physical system.
                      It should have attributes 'x' (position), 'v' (velocity), 
                      'm' (mass), and 'dt' (time step).
    - potential (object): An object representing the potential energy landscape.
                         It should have a 'force' method that calculates the force 
                         at a given position.
    
    Returns:
    None: The function modifies the 'x' and 'v' attributes of the provided system object in place.
    """
    # Calculate force
    force = potential.force(system.x, system.h)[0]
    
    # Update position and velocity using Euler method
    new_x = system.x + system.v * system.dt + (force/(2 * system.m)) * system.dt * system.dt
    new_v = system.v + (force/system.m) * system.dt
    
    # Update system state
    system.x = new_x
    system.v = new_v

    return None


def verlet_step(system, potential):
    """
    Perform a single Verlet integration step.
    
    Parameters:
    - system (object): An object representing the physical system.
                      It should have attributes 'x' (position), 'v' (velocity), 
                      'm' (mass), and 'dt' (time step).
    - potential (object): An object representing the potential energy landscape.
                         It should have a 'force' method that calculates the force 
                         at a given position.
    
    Returns:
    None: The function modifies the system object in place.
    """
    # Check if this is the first call by looking for x_previous attribute
    if not hasattr(system, "x_previous"):
        # First call: Initialize using Euler-like step
        force = potential.force(system.x, system.h)[0]
        
        # Store current position before updating it
        system.x_previous = system.x
        
        # First position using Euler step
        system.x = system.x + system.v * system.dt + (force/(2 * system.m)) * system.dt * system.dt
    else:
        # Regular Verlet step
        # Store current position and previous position
        x_current = system.x
        x_previous = system.x_previous
        
        # Calculate force at current position
        force = potential.force(x_current, system.h)[0]
        
        # Update position using Verlet algorithm
        system.x = 2 * x_current - x_previous + (force/system.m) * system.dt * system.dt
        
        # Update stored previous position for next step
        system.x_previous = x_current
        
        # Calculate velocity (central difference)
        system.v = (system.x - x_previous) / (2 * system.dt)
    
    return None


def leapfrog_step(system, potential):
    """
    Perform a single Leap Frog integration step.
    
    Parameters:
    - system (object): An object representing the physical system.
                      It should have attributes 'x' (position), 'v' (velocity), 
                      'm' (mass), 'dt' (time step), and 'v_half' (optional, velocity at half step).
    - potential (object): An object representing the potential energy landscape.
                         It should have a 'force' method that calculates the force 
                         at a given position.
    
    Returns:
    None: The function modifies the 'x', 'v', and 'v_half' attributes of the provided system object.
    """
    # Initialize v_half if not present
    if not hasattr(system, 'v_half'):
        # Calculate initial force
        force = potential.force(system.x, system.h)[0]
        system.v_half = system.v + (force/system.m) * (system.dt/2)
    
    # Calculate force at current position
    force = potential.force(system.x, system.h)[0]
    
    # Update velocity at half step
    system.v_half = system.v_half + (force/system.m) * system.dt
    
    # Update position using half-step velocity
    system.x = system.x + system.v_half * system.dt
    
    # Update velocity to full step for output/analysis
    # We could use the force at the new position for better accuracy, but we'd need another force calculation
    system.v = system.v_half - (force/system.m) * (system.dt/2)
    
    return None


def velocity_verlet_step(system, potential):
    """
    Perform a single Velocity Verlet integration step.
    
    Parameters:
    - system (object): An object representing the physical system.
                      It should have attributes 'x' (position), 'v' (velocity), 
                      'm' (mass), and 'dt' (time step).
    - potential (object): An object representing the potential energy landscape.
                         It should have a 'force' method that calculates the force 
                         at a given position.
    
    Returns:
    None: The function modifies the system object in place.
    """
    # Calculate force at current position (force01)
    force01 = potential.force(system.x, system.h)[0]
    
    # Update position using current velocity and force
    new_x = system.x + system.v * system.dt + 0.5 * force01/system.m * system.dt**2
    
    # Calculate force at new position (force02)
    force02 = potential.force(new_x, system.h)[0]
    
    # Update velocity using average force
    new_v = system.v + 0.5 * (force01 + force02)/system.m * system.dt
    
    # Update system state
    system.x = new_x
    system.v = new_v
    
    return None

#-----------------------------------------
#   T R A J E C T O R Y   I N T E G R A T O R S
#-----------------------------------------

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
        velocities[t + 1] = (velocity_halfsteps[t + 1] + 
                            velocity_halfsteps[t]) / 2
        
        # Update system state
        system.x = positions[t + 1]
        system.v = velocities[t + 1]
    
    return positions, velocities

def velocity_verlet(system, potential, n_steps):
    """
    Perform Velocity Verlet integration of Newton's equations of motion.
    
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
    
    # Main integration loop
    for t in range(n_steps):
        # Calculate force at current position (force01)
        force01 = potential.force(positions[t], system.h)[0]
        
        # Update position using current velocity and force
        positions[t + 1] = positions[t] + velocities[t] * system.dt + \
                          (force01/system.m) * system.dt * system.dt
        
        # Calculate force at new position (force02)
        force02 = potential.force(positions[t + 1], system.h)[0]
        
        # Update velocity using average force
        velocities[t + 1] = velocities[t] + \
                           ((force01 + force02)/(2 * system.m)) * system.dt
        
        # Update system state
        system.x = positions[t + 1]
        system.v = velocities[t + 1]
        
    return positions, velocities
