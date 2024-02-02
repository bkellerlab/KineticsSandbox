# 1-dimensional potentials

## Class D1

Class D1 is the parent class for all one-dimensional potentials. It provides the following a **abstract mehods** which need to be implemented for each specific potential $V(x)$.

- **\_\_init\_\_**:  set the parameters of the potential
- **potential**: expects $x$, returns $V(x)$, should be implemented as an analytical expression
- **force:** expects x, returns $-dV(x)/dx$, should be implemented as an analytical expression
- **hessian**: expects x, returns  $d^2V(x)/dx^2$, should be implemented as an analytical expression

The also class provides methods that are inherited by the child classes

- **negated_potential**: expects $x$, calls method potentials, returns $-V(x)$
- **force_num**: expects $x$ and $h$, calls method potentials, returns $-dV(x)/dx$ calculated via finite difference
- **hessian_num**: expects $x$ and $h$, calls method potentials, returns  $d^2V(x)/dx^2$ calculated via finite difference
- **min:** expects $x_{\mathrm{start}}$, returns location of the nearest minimum calculated via scipy.optimize
- **TS:** expects $x_{\mathrm{start}}$ and $x_{\mathrm{end}}$, returns location of the highest energy point in the intervall $[x_{\mathrm{start}}, x_{\mathrm{end}}]$$ (i.e. the transition state), calculated via scipy.optimize

For potentials that do not have minimum or a transition state, **min** and **TS** need to be overwritten so that they return an error.

## Linear potential

**Parameters:** $k, d$

**Range**: $x_{\mathrm{min}} = -\frac{d}{k} - 5$, $x_{\mathrm{max}} = -\frac{d}{k} + 5$ 
($\pm$ 5 nm from the zero point)

**Potential:**
$$V(x) = kx + d$$

**Force:**
$$F(x) = - \frac{d}{dx} V(x) = k$$
**Hessian:**
$$H_{11}(x) = \frac{d^2}{dx^2}V(x) = 0$$
**Transition state:** A linear potential does not have a transition state. Overwrite class function, return error.
## Harmonic potential

**Parameters:** $k, a, d$

**Range**: $x_{\mathrm{min}} = a - 5$, $x_{\mathrm{max}} = a + 5$ 
($\pm$ 5 nm from the minimum)

**Potential:**
$$V(x) = k (x-a)^2 + d$$
**Force:**
$$F(x) = - \frac{d}{dx} V(x) = 2k (x-a)$$

**Hessian:**
$$H_{11}(x) = \frac{d^2}{dx^2}V(x) = 2k$$

**Transition state:** A harmonic potential does not have a transition state. Overwrite class function, return error.
## Morse potential
## Gaussian bias
## Double well
## Bolhuis potential
## Triple well potential
## Prinz potential

# 2-dimensional potentials
# 3-dimensional potentials
# n-dimensional potentials

