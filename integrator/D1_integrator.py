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

