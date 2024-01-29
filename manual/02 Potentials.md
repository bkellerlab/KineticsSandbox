# 1-dimensional potentials

## Linear potential

**Parameters:** $k, d$
**Range**: $x_{\mathrm{min}} = -\frac{d}{k} - 5$, $x_{\mathrm{max}} = -\frac{d}{k} + 5$ 
($\pm$ 5 nm from the zero point)

**Potential:**
$$
	V(x) = kx + d
$$
**Force:**
$$
	F(x) = - \frac{d}{dx} V(x) = k
$$
**Hessian:**
$$
	H_{11}(x) = \frac{d^2}{dx^2}V(x) = 0
$$
**Transition state:** A linear potential does not have a transition state. Overwrite class function, return error.
## Harmonic potential

**Parameters:** $k, a, d$
**Range**: tbd

**Potential:**
$$
	V(x) = k (x-a)^2 + d
$$
**Force:**
$$
	F(x) = - \frac{d}{dx} V(x) = 2k (x-a)
$$
**Hessian:**
$$
	H_{11}(x) = \frac{d^2}{dx^2}V(x) = 2k
$$
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

