# Transition state theory

The rate constant in Eyring transition state theory is given by
$$k_{AB}^{\mathrm{Eyr}} = \frac{RT}{h} \frac{\widetilde{q}_{AB^\ddagger, \mathrm{vib}}}{q_{A, \mathrm{vib}}} \cdot \frac{q_{AB^\ddagger, \mathrm{rot}}}{q_{A, \mathrm{rot}}} \cdot \frac{q_{AB^\ddagger, \mathrm{trans}}}{q_{A, \mathrm{trans}}} \cdot \exp\left(- \frac{E_b}{RT} \right)$$
where 
- $q_{A, \mathrm{trans}}$ and $q_{AB^\ddagger, \mathrm{trans}}$  are the translational partition function at the minimum $A$ and at the transition state
-  $q_{A, \mathrm{rot}}$  and $q_{AB^\ddagger, \mathrm{rot}}$ are the rotational partition function at the minimum $A$ and at the transition state
-  $q_{A, \mathrm{vib}}$ and $\widetilde{q}_{AB^\ddagger, \mathrm{vib}}$  are the vibrational partition function at the minimum $A$ and at the transition state. The tilde indicates that the reactive vibration has been excluded from the vibrational partition function of the transition state.
- $E_b$ is the energy difference between the minimum and the transition state

In low-dimensional systems, there is now rotational and translational partition function, and the expression for the rate constant reduces to
$$k_{AB}^{\mathrm{Eyr}} = \frac{RT}{h} \frac{\widetilde{q}_{AB^\ddagger, \mathrm{vib}}}{q_{A, \mathrm{vib}}}\cdot \exp\left(- \frac{E_b}{RT} \right)$$
### One-dimensional potentials

The one-dimensional potential $V(x)$ is expanded around an extremum at $x=x_0$ in a Taylor series
$$
\begin{align}
V(x)\mid_{x=x_0} &= V(x_0) + \frac{d}{d_x}V(x)\mid_{x=x_0} (x-x_0) +\frac{1}{2}\frac{d^2}{dx^2}V(x)\mid_{x=x_0} (x-x_0)^2 + \dots\cr
&= V(x_0) + \frac{1}{2}k (x-x_0)^2 + \dots\cr
\end{align}
$$
with 
$$
	k = \frac{d^2}{dx^2}V(x)\mid_{x=x_0}
$$
The quantum mechanical eigenstates of this harmonic potential are
$$
	\epsilon_n = h \nu  \left(n + \frac{1}{2}\right) = \frac{h}{2\pi}\sqrt{\frac{k}{m}}\left(n + \frac{1}{2}\right)
$$
where the frequency is given 
$$
	\nu = \frac{1}{2\pi}\sqrt{\frac{k}{m}}
$$
The vibrational partition function is
$$
	q_{A, \mathrm{vib}} 
	= \frac{\exp\left(-\frac{h \nu}{2RT}\right)}{1 - \exp\left(-\frac{h \nu}{RT}\right)}
$$
$$q_{AB^\ddagger, \mathrm{vib}} = 1$$ 