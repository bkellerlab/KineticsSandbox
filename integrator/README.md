# Deterministic Integrators

This section describes the deterministic integrators implemented in the `integrator.D1_integrator` module for one-dimensional systems.

## Overview

The module implements three classical molecular dynamics integrators:
- Euler integrator
- Verlet integrator
- Leap-frog integrator

Each integrator solves Newton's equations of motion:

$$
\begin{align}
\dot{x} &= v \\
\dot{v} &= F(x)/m
\end{align}
$$

where $x$ is position, $v$ is velocity, $F(x)$ is the force, and $m$ is mass.

## Euler Integrator

The Euler method is the simplest numerical integration scheme. It updates positions and velocities using:

$$
\begin{align}
x_{k+1} &= x_k + v_k\Delta t + \frac{F(x_k)}{2m}(\Delta t)^2 \\
v_{k+1} &= v_k + \frac{F(x_k)}{m}\Delta t
\end{align}
$$

While simple to implement, it has relatively poor energy conservation properties for long simulations.

## Verlet Integrator

The Verlet algorithm is a more sophisticated method that provides better energy conservation. It uses the positions from two previous timesteps:

$$
\begin{align}
x_{k+1} &= 2x_k - x_{k-1} + \frac{F(x_k)}{m}(\Delta t)^2 \\
v_k &= \frac{x_{k+1} - x_{k-1}}{2\Delta t}
\end{align}
$$

The velocities are computed using central differences, which provides better accuracy than the Euler method.

## Leap-frog Integrator

The leap-frog algorithm updates positions and velocities at interleaved time points, with velocities computed at half-steps:

$$
\begin{align}
v_{k+1/2} &= v_{k-1/2} + \frac{F(x_k)}{m}\Delta t \\
x_{k+1} &= x_k + v_{k+1/2}\Delta t
\end{align}
$$

The full-step velocities are computed as averages of half-step values:

$$
v_k = \frac{v_{k+1/2} + v_{k-1/2}}{2}
$$

This scheme provides good energy conservation and is time-reversible.

## Usage

All integrators follow the same interface:

```python
positions, velocities = integrator(system, potential, n_steps)
```

where:
- `system`: System object with attributes for mass, position, velocity, and timestep
- `potential`: Potential energy object with a force method
- `n_steps`: Number of integration steps to perform

The integrators return arrays containing the position and velocity trajectories.

Example usage:

```python
import numpy as np
from system import system
from potential import D1
from integrator import D1_integrator

# Initialize system
sys = system.D1(m=100.0, x=0.0, v=0.0, T=300.0, xi=1.0, dt=0.001, h=0.001)

# Initialize potential (e.g., harmonic)
pot = D1.Quadratic([1.0, 0.0])  # k=1.0, x0=0.0

# Run simulation
n_steps = 1000
positions, velocities = D1_integrator.verlet(sys, pot, n_steps)
```



