#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 2025
Last modified on Tue May 06 2025

@author: Ahmet Sarigun

This module implements integration algorithms for N-dimensional systems.
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
    Perform a single Euler integration step for Newton's equations of motion
    in N dimensions.
    
    Parameters:
    - system (object): Must have attributes
          x    : position array, shape (..., D)
          v    : velocity array, shape (..., D)
          m    : mass (scalar) or array broadcastable to x/v
          dt   : time step (scalar)
          h    : optional parameter passed to potential.force
    - potential (object): Must implement
          force(x, h) -> array of same shape as x
    
    Updates system.x and system.v in place.
    """
    # Compute N-D force array
    force = potential.force(system.x, system.h)
    
    
    new_x = system.x + system.v * system.dt + (force / (2 * system.m)) * system.dt**2
    new_v = system.v + (force / system.m) * system.dt

    # Update system state
    system.x = new_x
    system.v = new_v

    return None


def verlet_step(system, potential):
    """
    Perform a single Verlet integration step in N dimensions.

    Parameters:
    - system (object): Must have attributes
          x    : position array, shape (..., D)
          v    : velocity array, shape (..., D)
          m    : mass (scalar) or array broadcastable to x/v
          dt   : time step (scalar)
          h    : optional parameter passed to potential.force
    - potential (object): Must implement
          force(x, h) -> array of same shape as x

    Updates system.x, system.x_previous, and system.v in place.
    Returns None.
    """

    if not hasattr(system, "x_previous"):
        # First call: Initialize using Euler-like step
        force = potential.force(system.x, system.h)

        # Store current position before updating it
        system.x_previous = np.copy(system.x)

        # First position using Euler step
        system.x = system.x + system.v * system.dt + (force / (2.0 * system.m)) * system.dt * system.dt

    else:
        # Regular Verlet step
        # Store current position and previous position
        x_curr = system.x
        x_prev = system.x_previous

        # Calculate force at current position
        force = potential.force(x_curr, system.h)

        # Update position using Verlet algorithm
        new_x = 2 * x_curr - x_prev + (force / system.m) * system.dt * system.dt
        new_v = (new_x - x_prev) / (2.0 * system.dt)

        system.x_previous = np.copy(x_curr)
        system.x = new_x
        system.v = new_v

    return None


def leapfrog_step(system, potential):
    """
    Perform a single Leap Frog integration step in N dimensions.
    
    Parameters:
    - system (object): Must have attributes
          x    : position array, shape (..., D)
          v    : velocity array, shape (..., D)
          m    : mass (scalar) or array broadcastable to x/v
          dt   : time step (scalar)
          h    : optional parameter passed to potential.force
    - potential (object): Must implement
          force(x, h) -> array of same shape as x

    Updates system.x, system.v_half, and system.v in place.
    Returns None.
    """
    # Initialize v_half if not present
    if not hasattr(system, 'v_half'):
        # Calculate initial force
        force = potential.force(system.x, system.h)
        system.v_half = system.v + (force / system.m) * (system.dt/2)
    
    # Calculate force at current position
    force = potential.force(system.x, system.h)
    
    # Update velocity at half step
    system.v_half = system.v_half + (force / system.m) * system.dt
    
    # Update position using half-step velocity
    system.x = system.x + system.v_half * system.dt
    
    # Update velocity to full step for output/analysis
    system.v = system.v_half - (force / system.m) * (system.dt/2)
    
    return None


def velocity_verlet_step(system, potential):
    """
    Perform a single Velocity Verlet integration step in n dimensions.
    
    - system (object): Must have attributes
          x    : position array, shape (..., D)
          v    : velocity array, shape (..., D)
          m    : mass (scalar) or array broadcastable to x/v
          dt   : time step (scalar)
          h    : optional parameter passed to potential.force
    - potential (object): Must implement
          force(x, h) -> array of same shape as x
    
    Returns:
    None: The function modifies the system object in place.
    """
    # Calculate force at current position (force01)
    force01 = potential.force(system.x, system.h)
    
    # Update position using current velocity and force
    new_x = system.x + system.v * system.dt + 0.5 * force01/system.m * system.dt**2
    
    # Calculate force at new position (force02)
    force02 = potential.force(new_x, system.h)
    
    # Update velocity using average force
    new_v = system.v + 0.5 * (force01 + force02)/system.m * system.dt
    
    # Update system state
    system.x = new_x
    system.v = new_v
    
    return None
