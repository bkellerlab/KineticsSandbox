# Langevin splitting integrators

## One-dimensional potentials
The equations for the individual steps of a Langevin splitting operator, when the state is formulated in terms of coordinates $q_k$ and associated momenta $p_k$, are

$$
\begin{align}
\mathcal{A} \left(\begin{array}{c} q_{k}\\p_{k}\end{array}\right) 
&= \left(\begin{array}{c} q_k+a p_k \\p_k\end{array}\right) \\
%
\mathcal{B}\left(\begin{array}{c} q_{k}\\p_{k}\end{array}\right) 
     &= \left(\begin{array}{c}q_k \\ p_k+b(q_k)\end{array}\right) \\ 
%
\mathcal{O}\left(\begin{array}{c} q_{k}\\ p_{k}\end{array}\right) 
     &= \left(\begin{array}{c} q_k \\ d \, p_k + f\, \eta_k\end{array}\right)
\end{align}
$$

When formulated in terms of positions $x_k$ and associated velocities $v_k$, they are
$$
\begin{align}
\mathcal{A} \left(\begin{array}{c} x_{k}\\v_{k}\end{array}\right) 
&= \left(\begin{array}{c} x_k+a v_k \\v_k\end{array}\right) \\
%
\mathcal{B}\left(\begin{array}{c} x_{k}\\v_{k}\end{array}\right) 
     &= \left(\begin{array}{c}x_k \\ v_k+b(x_k)\end{array}\right) \\ 
%
\mathcal{O}\left(\begin{array}{c} x_{k}\\ v_{k}\end{array}\right) 
     &= \left(\begin{array}{c} x_k \\ d \, v_k + f_v\, \eta_k\end{array}\right)
\end{align}
$$
with coefficients
$$
\begin{align}
a      &= \Delta t \,\frac{1}{m}\\
b(q_k) &= \Delta t \,F(q_k)\\
b(x_k) &= \Delta t \,F(x_k)\\ \\
d      &= e^{-\xi\Delta t}\\
f      &= \sqrt{RTm\,(1-e^{-2\xi\Delta t})}\\
f_v   &= \frac{1}{m}\sqrt{RTm\,(1-e^{-2\xi\Delta t})} = \sqrt{\frac{RT\,(1-e^{-2\xi\Delta t})}{m}} 
\end{align}
$$
The force is given as 
$$
F(q_k) = -\Delta t\nabla V(q_k) \, .
$$
$R$ is the ideal gas constant in units $\mathrm{kJ}\,\mathrm{mol}^{-1}\, \mathrm{K}^{-1}$.
If a step is applied with half a time-step, the corresponding operator is denoted with a prime, i.e. $\mathcal{A}'$,$\mathcal{B}'$, $\mathcal{O}'$. 

The system class holds the state of the system and parameters of the dynamics
- $x_k$ position
- $v_k$ velocity
- $m$ mass
- $\Delta t$ time step
- $T$ temperature
- $\xi$ friction coefficient

## Update operators for various splitting algorithms
#### ABO
The ABO algorithm implements the update operator
$$
\left(\begin{array}{c} x_{k+1}\\v_{k+1}\end{array}\right) =
\mathcal{O}\mathcal{B}\mathcal{A} \left(\begin{array}{c} x_{k}\\v_{k}\end{array}\right) 
$$
#### ABOBA
The ABO algorithm implements the update operator
$$
\left(\begin{array}{c} x_{k+1}\\v_{k+1}\end{array}\right) =
\mathcal{A}'\mathcal{B}'\mathcal{O}\mathcal{B}'\mathcal{A}' \left(\begin{array}{c} x_{k}\\v_{k}\end{array}\right) 
$$
#### AOBOA
$$
\left(\begin{array}{c} x_{k+1}\\v_{k+1}\end{array}\right) =
\mathcal{A}'\mathcal{O}'\mathcal{B}\mathcal{O}'\mathcal{A}' \left(\begin{array}{c} x_{k}\\v_{k}\end{array}\right) 
$$
#### BAOAB
$$
\left(\begin{array}{c} x_{k+1}\\v_{k+1}\end{array}\right) =
\mathcal{B}'\mathcal{A}'\mathcal{O}\mathcal{A}'\mathcal{B}' \left(\begin{array}{c} x_{k}\\v_{k}\end{array}\right) 
$$

#### BOAOB
$$
\left(\begin{array}{c} x_{k+1}\\v_{k+1}\end{array}\right) =
\mathcal{B}'\mathcal{O}'\mathcal{A}\mathcal{O}'\mathcal{B}' \left(\begin{array}{c} x_{k}\\v_{k}\end{array}\right) 
$$
#### OBABO
$$
\left(\begin{array}{c} x_{k+1}\\v_{k+1}\end{array}\right) =
\mathcal{O}'\mathcal{B}'\mathcal{A}\mathcal{B}'\mathcal{O}' \left(\begin{array}{c} x_{k}\\v_{k}\end{array}\right) 
$$
#### OABAO
$$
\left(\begin{array}{c} x_{k+1}\\v_{k+1}\end{array}\right) =
\mathcal{O}'\mathcal{A}'\mathcal{B}\mathcal{A}'\mathcal{O}' \left(\begin{array}{c} x_{k}\\v_{k}\end{array}\right) 
$$
#### BAOA
$$
\left(\begin{array}{c} x_{k+1}\\v_{k+1}\end{array}\right) =
\mathcal{A}'\mathcal{O}\mathcal{A}'\mathcal{B} \left(\begin{array}{c} x_{k}\\v_{k}\end{array}\right) 
$$
