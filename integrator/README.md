# Deterministic Integrators

This section describes the deterministic integrators implemented in the `integrator.D1_integrator` and `integrator.Dn_integrator` modules for one-dimensional and N-dimensional systems respectively.

## Overview

The modules implement classical molecular dynamics integrators that solve Newton's equations of motion:

**For 1D systems:**
$$
\begin{align}
\dot{x} &= v \\
\dot{v} &= F(x)/m
\end{align}
$$

**For N-dimensional systems:**
$$
\begin{align}
\dot{\mathbf{x}} &= \mathbf{v} \\
\dot{\mathbf{v}} &= \mathbf{F}(\mathbf{x})/m
\end{align}
$$

where $\mathbf{x}$ is the position vector, $\mathbf{v}$ is the velocity vector, $\mathbf{F}(\mathbf{x})$ is the force vector, and $m$ is mass.

## One-Dimensional Integrators (D1_integrator)

### Euler Integrator

The Euler method is the simplest numerical integration scheme. It updates positions and velocities using:

$$
\begin{align}
x_{k+1} &= x_k + v_k\Delta t + \frac{F(x_k)}{2m}(\Delta t)^2 \\
v_{k+1} &= v_k + \frac{F(x_k)}{m}\Delta t
\end{align}
$$

While simple to implement, it has relatively poor energy conservation properties for long simulations.

### Verlet Integrator

The Verlet algorithm is a more sophisticated method that provides better energy conservation. It uses the positions from two previous timesteps:

$$
\begin{align}
x_{k+1} &= 2x_k - x_{k-1} + \frac{F(x_k)}{m}(\Delta t)^2 \\
v_k &= \frac{x_{k+1} - x_{k-1}}{2\Delta t}
\end{align}
$$

The velocities are computed using central differences, which provides better accuracy than the Euler method.

### Leap-frog Integrator

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

### Velocity Verlet Integrator

The Velocity Verlet algorithm is a symplectic integrator that provides excellent energy conservation. It updates positions and velocities using:

$$
\begin{align}
x_{k+1} &= x_k + v_k\Delta t + \frac{F(x_k)}{2m}(\Delta t)^2 \\
v_{k+1} &= v_k + \frac{F(x_k) + F(x_{k+1})}{2m}\Delta t
\end{align}
$$

Unlike the standard Verlet algorithm, Velocity Verlet explicitly computes velocities at each step, making it more convenient for analysis and visualization. It's particularly well-suited for molecular dynamics simulations due to its stability and energy conservation properties.

## N-Dimensional Integrators ($D_n$ Integrator)

The N-dimensional integrators extend the 1D algorithms to handle multi-dimensional systems where positions and velocities are vectors. All algorithms maintain the same mathematical foundations but operate on vector quantities.

### N-D Euler Integrator

Extends the Euler method to N dimensions:

$$
\begin{align}
\mathbf{x}_{k+1} &= \mathbf{x}_k + \mathbf{v}_k\Delta t + \frac{\mathbf{F}(\mathbf{x}_k)}{2m}(\Delta t)^2 \\
\mathbf{v}_{k+1} &= \mathbf{v}_k + \frac{\mathbf{F}(\mathbf{x}_k)}{m}\Delta t
\end{align}
$$

**System requirements:**
- `system.x`: position array of shape `(..., D)` where D is the number of dimensions
- `system.v`: velocity array of shape `(..., D)`
- `system.m`: mass (scalar or array broadcastable to x/v)
- `system.dt`: time step (scalar)
- `system.h`: parameter passed to potential.force

### N-D Verlet Integrator

Extends the Verlet algorithm to N dimensions:

$$
\begin{align}
\mathbf{x}_{k+1} &= 2\mathbf{x}_k - \mathbf{x}_{k-1} + \frac{\mathbf{F}(\mathbf{x}_k)}{m}(\Delta t)^2 \\
\mathbf{v}_k &= \frac{\mathbf{x}_{k+1} - \mathbf{x}_{k-1}}{2\Delta t}
\end{align}
$$

**Additional system requirements:**
- `system.x_previous`: stores the previous position for the Verlet algorithm
- Automatically initialized on first call using an Euler-like step

### N-D Leap-frog Integrator

Extends the leap-frog algorithm to N dimensions:

$$
\begin{align}
\mathbf{v}_{k+1/2} &= \mathbf{v}_{k-1/2} + \frac{\mathbf{F}(\mathbf{x}_k)}{m}\Delta t \\
\mathbf{x}_{k+1} &= \mathbf{x}_k + \mathbf{v}_{k+1/2}\Delta t \\
\mathbf{v}_k &= \mathbf{v}_{k+1/2} - \frac{\mathbf{F}(\mathbf{x}_k)}{m}\frac{\Delta t}{2}
\end{align}
$$

**Additional system requirements:**
- `system.v_half`: stores the half-step velocity
- Automatically initialized on first call

### N-D Velocity Verlet Integrator

Extends the Velocity Verlet algorithm to N dimensions:

$$
\begin{align}
\mathbf{x}_{k+1} &= \mathbf{x}_k + \mathbf{v}_k\Delta t + \frac{\mathbf{F}(\mathbf{x}_k)}{2m}(\Delta t)^2 \\
\mathbf{v}_{k+1} &= \mathbf{v}_k + \frac{\mathbf{F}(\mathbf{x}_k) + \mathbf{F}(\mathbf{x}_{k+1})}{2m}\Delta t
\end{align}
$$

This is the most robust algorithm for N-dimensional systems, providing excellent energy conservation and stability.

## Key Differences: 1D vs N-D Implementations

| Aspect | 1D Implementation | N-D Implementation |
|--------|-------------------|-------------------|
| **Data Types** | Scalars for position/velocity | Arrays for position/velocity vectors |
| **Force Calculation** | Returns scalar force | Returns force vector of same shape as position |
| **Mass Handling** | Scalar mass | Scalar or array broadcastable to position/velocity |
| **Memory Management** | Simple scalar operations | Array operations with proper shape handling |
| **Initialization** | Direct scalar assignment | Array copying with `np.copy()` |

## Potential Interface Requirements

For both 1D and N-D integrators, the potential object must implement:

```python
def force(self, x, h):
    """
    Calculate force at position x.
    
    Parameters:
    - x: position (scalar for 1D, array for N-D)
    - h: step size parameter
    
    Returns:
    - Force (scalar for 1D, array of same shape as x for N-D)
    """
```

## Usage

You can find usage examples in the `cookbooks/D1_integrator.ipynb` file.

For testing the integrators:

```bash
# Test 1D integrators
python test_D1_integrators.py

# Test N-D integrators 
python test_Dn_integrators.py
```

You can find the results in the [WandB project](https://wandb.ai/asarigun/integrator-comparison).

## Performance Considerations

- **1D integrators** are optimized for scalar operations and are fastest for single-particle systems
- **N-D integrators** handle multi-dimensional systems efficiently using vectorized operations
- **Velocity Verlet** is generally recommended for both 1D and N-D systems due to its superior stability and energy conservation
- **Euler** should be avoided for long simulations due to energy drift
- **Verlet** and **Leap-frog** provide good energy conservation with moderate computational cost

