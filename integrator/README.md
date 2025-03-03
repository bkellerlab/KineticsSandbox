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

## Velocity Verlet Integrator

The Velocity Verlet algorithm is a symplectic integrator that provides excellent energy conservation. It updates positions and velocities using:

$$
\begin{align}
x_{k+1} &= x_k + v_k\Delta t + \frac{F(x_k)}{m}(\Delta t)^2 \\
v_{k+1} &= v_k + \frac{F(x_k) + F(x_{k+1})}{2m}\Delta t
\end{align}
$$

Unlike the standard Verlet algorithm, Velocity Verlet explicitly computes velocities at each step, making it more convenient for analysis and visualization. It's particularly well-suited for molecular dynamics simulations due to its stability and energy conservation properties.

## Usage

You can find the usage examples in the `cookbooks/D1_integrator.ipynb` file.

Also, you can run the test script to see the performance of the integrators:

```bash
python test.py
```
You can find the results in the [WandB project](https://wandb.ai/asarigun/integrator-comparison).

