#%%
#-----------------------------------------------------------------
# This script shows how to use the updated modules o run a biased 
# simulation and reweighting on the fly.
#
# UPDATES:
# - stochastic integrator 
#       D1_stochastic.py: langevin integrator functions include 
#                          option for simulating on a biased 
#                          potential and reweighting factor generation 
#                          NOTE not ideal solution, would be better in 
#                          potential class, as we can than also change 
#                          sign of force output -> important for reweighting
#                          with pertubation potential      
# - system class 
#       system.py: include arguments for the bias force class
#                  and the eta_k, logM at current integration step), 
# 
# NOTE:
# - call of random number and bias force at the correct update step is 
# managed in the stochastic integrator functions in D1_stochastic.py 
# - path reweighting factor logM is written to system
# 
# TODO:
# - update potential module for Bolhuis_Bias
# - create module for simulation functions like ABOBA_simulation()
#-----------------------------------------------------------------

import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt

from system import system
from potential import D1 as pot
from integrator import D1_stochastic as sd
from utils import thermodynamics as thermo

#-----------------------------------------------------------------
# define bias 
# here we just give a bias class to fullfill set up in biased integrator:
# if bias is not None:
#        force = potential.force_num( system.x, 0.001 )[0] + bias.force_ana( system.x )[0]
#    else:
#        force = potential.force_num( system.x, 0.001 )[0] 
#-----------------------------------------------------------------
class Bolhuis_Bias(): 
    def __init__(self, param): 
        """Bias potential for Bolhuis potential.

        Parameters:
            a (float) - parameter controlling the center of the quadratic term.
            c (float) - parameter controlling the width of perturbation.
            alpha (float) - strength of the perturbation.
        Raises:
        - ValueError: If param does not have exactly e elements.
        """
        
        # Check if param has the correct number of elements
        if len(param) != 3:
            raise ValueError("param must have exactly 6 elements.")
        
        # Assign parameters
        self.a = param[0]
        self.c = param[1]
        self.alpha = param[2]
        
    # the potential energy function 
    def potential(self, x):
        """
        The potential energy function is given by:
        V(x) = alpha * np.exp(-c * (x - a)**2)
    
        The units of V(x) are kJ/mol, following the convention in GROMACS.
    
        Parameters:
            - x (float): position

        Returns:
            float: The value of the potential energy function at the given position x.
        """
    
        return self.alpha * np.exp(-self.c * (x - self.a)**2)
          
    # the force, analytical expression 
    def force_ana(self, x):
        """
        The force is given by:
        F(x) = - dV(x) / dx 
             = alpha * np.exp(-c * (x - a)**2) * c * 2 * (x - a)
    
        The units of F(x) are kJ/(mol * nm), following the convention in GROMACS.
    
        Parameters:
            - x (float): position
    
        Returns:
            numpy array: The value of the force at the given position x, returned as vector with 1 element.
        """
        
        F = self.alpha * np.exp(-self.c * (x - self.a)**2) * self.c * 2 * (x - self.a)
        return np.array([F])


#-----------------------------------------------------------------
# define simulation schema 
#-----------------------------------------------------------------
def ABOBA_simulation(system, potential, bias_class, n_steps, n_steps_out):
    '''To perform ABOBA simulation and capture reweighting factors.
    Args:
    system (system.D1): system class capturing current state
    potential (pot.D1): unbiased simulation potential class including potential and force
    bias (pot.D1): biased potential class including potential and force
    n_steps (int): number of simulation steps
    n_steps_out (int): write out every n_steps_out NOTE we need to consider for sum over path in case of logM
    Return
    pos (array): x trajectory
    g_factor (array): bias potential weighted by thermal energy NOTE sign and exp
    M_factor (array): path reweighting pre-factor NOTE sign and exp
    '''
    # frequency at which progress is written to the terminal
    n_steps_report =  n_steps // 5

    # initialize arrays NOTE I do not save the initial step (as openMM)
    pos = np.zeros(n_steps)     
    g_factor = np.zeros(n_steps)
    M_factor = np.zeros(n_steps)

    bias_force = bias_class.force_ana
    # loop over simulation steps
    for k in range(n_steps):
    
        # report progress
        if k % n_steps_report == 0:
            print('Simulation progress: ',
                   k/n_steps * 100, '%')
    
        # perform an biased integration step with Girsanov reweighting
        sd.ABOBA(system, potential, bias_force, path_reweighting=True)
        
        # NOTE here we need to introduce sum over path for logM if we use n_steps_out
        pos[k] = system.x
            
        # reweighting factors
        g_factor[k] = bias_class.potential(system.x) / (const.R * 0.001 * system.T)
        M_factor[k] = system.logM
    
    return pos, g_factor, M_factor

#-----------------------------------------------------------------
# define system
#-----------------------------------------------------------------
m  = 1.0    
x = 0.0
v = 0.0 
T = 300   # K    
xi = 2.5  # 1/ps     
dt = 0.1  # ps == 100 fs

#-----------------------------------------------------------------
# define unbiased potential
#-----------------------------------------------------------------
a=2
b=1
c=0
k1=10
k2=2
alpha=0
param = [a, b, c, k1, k2, alpha]
potential = pot.Bolhuis(param)

#-----------------------------------------------------------------
# simulation + sampled Boltzman vs analytical Boltzmann
# simulation potential is potential + bias 
# --> that is added in the B integration step
# --> here we just input unbiased potential and bias (created in loop)
#-----------------------------------------------------------------
# MD parameter
n_steps = 1_000_000 
n_steps_out = 1
# analysis 
x_bounds_min, x_bounds_max = .0, 4.0
x_line = np.linspace(x_bounds_min, x_bounds_max, 401)
dx = x_line[1]-x_line[0]

alpha_range = np.arange(2, 8, 2) #10, 2)

distribution_difference_mean_alpha, distribution_difference_std_alpha = [], []

for alpha in alpha_range:
    # change bias for barrier hight: 
    # param[0]: a = 2.0 - param[1]: c = 20.0  - param[2]: alpha
    param=[2.0, 20, alpha]
    bias = Bolhuis_Bias(param) 
    # initialize the system
    simulation_system = system.D1(m, x, v, T, xi, dt, h=1, bias_force=bias.force_ana)
    # set boltzmann ditribution 
    potential_ref = pot.Bolhuis([a, b, 20, k1, k2, alpha])
    x_val = x_line[0:-1] + ( dx /2 )
    boltzmann_density = thermo.boltzmann_factor(potential_ref, x_val, simulation_system.T)
    boltzmann_density = boltzmann_density / (np.sum(boltzmann_density) * dx)
    # run simulation
    difference = []
    for i in range(3):
        pos, g_factor, M_factor = ABOBA_simulation(
            simulation_system, 
            potential, 
            bias, 
            n_steps,
            n_steps_out
            )

        # get a histogram of the sampled density
        counts, _ = np.histogram(pos, bins=x_line, density=True)
        distribution_difference = counts - boltzmann_density
        difference.append(distribution_difference)

    # give mean difference to boltzmann distribution
    distribution_difference_mean = np.mean(difference, axis=0)
    distribution_difference_std = np.std(difference, axis=0)
    distribution_difference_mean_alpha.append(distribution_difference_mean)
    distribution_difference_std_alpha.append(distribution_difference_std)
#%%    
plt.figure(figsize=(12, 6))

cmap = plt.get_cmap('viridis')
colors = cmap(np.linspace(0, 1, len(alpha_range)))  # evenly spaced colors

for color, alpha, distribution_difference_mean, distribution_difference_std in zip(
    colors, alpha_range, distribution_difference_mean_alpha, distribution_difference_std_alpha
):    
    plt.plot(
        x_val, 
        distribution_difference_mean, 
        color=color, 
        label=rf'$\alpha$={alpha}', 
        linewidth=3
    )
    
    plt.fill_between(
        x_val, 
        distribution_difference_mean - distribution_difference_std, 
        distribution_difference_mean + distribution_difference_std, 
        color=color, 
        alpha=0.2
    )

plt.xlabel("x in nm", fontsize=20)
plt.ylabel(r"$P_{\rm sampled}(x) - P_{\rm Boltzmann}(x)$ in kJ/mol", fontsize=20)
plt.legend(fontsize=16)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

# %%