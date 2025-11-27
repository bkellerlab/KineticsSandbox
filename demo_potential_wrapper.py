#---------------------------------------------------------------------
# This script shows possible modification in modules:
# - potenntial/D1.py
# - integrator/D1_stochastic_integrator.py
#
# Goal:
# enable reweighting for U is bias or pertubation potential
#
# Problem:
# potential modification managed in integrator 
# -> we can not give pertubation potential (e.g. U=TW-DW) without 
#    modify simulation potential
# -> we can not bias the input potential with +U and calculate 
#    reweighting factors with -U
#
# Idea:
# decouple potential modification from integration
#   => we create two wrapper calsses 
#       - BiasedPotential to combine base potential with modulation potential
#       - PertubationPotential to get difference between two analytical function
#   => input to update function correct simulation potential
#   => modify B_step delete simulation potential modification
#
# Effect:
#   -> simulation potential -> can be provided by BiasedPotential
#   -> bias class can be input to BiasedPotential and integration 
#      class with different sign
#   -> pertubation dont need to be defined for simulation potential
#      but could be for reweighting 
#---------------------------------------------------------------------
#%%
import numpy as np
from potential import D1 as pot
import matplotlib.pyplot as plt
from pathlib import Path
from system import system
from integrator import D1_stochastic as sd
# x range
x_line = np.linspace(-.58, .58, 401)
dx = x_line[1]-x_line[0]
x_val = x_line[0:-1] + ( dx /2 )
color_target="darkgreen"
color_simulation="darkblue"

# %% 
#---------------------------------------------------------------------           
# D E M O :  Reweighting with pertubation potential 
#---------------------------------------------------------------------
# define simulation potential 
param_dw = [
    195,    # k kJ/(mol nm^4) 
    0,      # a nm 
    0.16    # b nm^2
]
simulation_potential = pot.DoubleWell(param_dw)
# define target potential
param_tw =[
    0.0,        # a  nm
    3.049,      # c1 kJ/(mol nm)
    83.7,       # c2 kJ/(mol nm^2)
    -28.36,     # c3 kJ/(mol nm^3)
    -1038.8,    # c4 kJ/(mol nm^4)
    0.0,        # c5 kJ/(mol nm^5)
    3078.8      # c6 kJ/(mol nm^6)
]
target_potential = pot.Polynomial(param_tw)
# define pertubation potential
pertubation_potential = pot.PertubationPotential(target_potential, simulation_potential)
#
# perform a simulation step k
simulation_system = system.D1(
    m = 40,     # 0.04 kg/mol -> 40 u; like Ca 
    x = 0.0,    # nm
    v = 0.0,    # nm/ps 
    T = 300,    # K  
    xi = 1,     # 1/ps     
    dt = 0.01,  # ps  
    h=0.0001
    )
for k in range(100):
    sd.ABO(
        system=simulation_system, 
        potential=simulation_potential, 
        bias=pertubation_potential, 
        girsanov_reweighting=True
        )
    force = pertubation_potential.force(simulation_system.x, simulation_system.h)
    
    print(np.equal(force[0],simulation_system.bias_force)) # only for ABO -> update and x in system the same

# plot system with pertubation -> output fig1
fig, (ax,ax2) = plt.subplots(2,1,figsize=(6.25, 6.75), sharex=True)
ax.plot(
    x_val, simulation_potential.potential(x_val),
    color=color_simulation, lw=3, label='simulation_potential'
    ) 
ax.plot(
    x_val, target_potential.potential(x_val),
    color=color_target, lw=3, label='target_potential'
    )
ax2.plot(
    x_val,  target_potential.potential(x_val)- simulation_potential.potential(x_val),
    color='k', lw=3. , label='target_potential- simulation_potential'
    ) 
ax2.plot(
    x_val,  pertubation_potential.potential(x_val), '--',
    color='g', lw=3, label='petrubation_potential'
    )
ax.set_ylim(-2, 5.5)
ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
ax.set_ylabel("potential", fontsize=20)
ax.tick_params(axis='both', labelsize=20)
ax.legend()
ax.grid() 
ax2.set_ylim(-5.75, 5.5)
ax2.set_xlabel("x", fontsize=20)
ax2.set_ylabel("pertibation", fontsize=20)
ax2.tick_params(axis='both', labelsize=20)
ax2.legend()
ax2.grid()

# %% 
#---------------------------------------------------------------------           
# D E M O :  Reweighting with bias potential 
#---------------------------------------------------------------------
# %% base potential 
param_dw = [
    195,    # k kJ/(mol nm^4) 
    0,      # a nm 
    0.16    # b nm^2
]
dw_potential = pot.DoubleWell(param_dw)
# bias potential
param_bias=[.0, 80, 2]
bias_potential = pot.GeneralGaussian(param_bias)
# simulation potential 
simulation_potential  = pot.BiasedPotential(
    potential_class=dw_potential,
    bias_class=bias_potential
    ) 
#
# perform a simulation step k
simulation_system = system.D1(
    m = 40,     # 0.04 kg/mol -> 40 u; like Ca 
    x = 0.0,    # nm
    v = 0.0,    # nm/ps 
    T = 300,    # K  
    xi = 1,     # 1/ps     
    dt = 0.01,  # ps  
    h=1
    )
for k in range(100):
    sd.ABO(
        system=simulation_system, 
        potential=simulation_potential, 
        bias=bias_potential, 
        girsanov_reweighting=True
        )
    print(
        simulation_system.bias_force == bias_potential.force(simulation_system.x, simulation_system.h)[0]
        )

# plot system with bias output fig 2
fig, (ax) = plt.subplots(figsize=(6.25, 6.75))

ax.plot(
    x_val, simulation_potential.potential(x_val),
         color=color_simulation, lw=3, label='simulation_potential'
         ) 

ax.plot(
    x_val, dw_potential.potential(x_val),
    color='k', lw=3, label='base_potential'
    )
ax.plot(
    x_val, bias_potential.potential(x_val),
    color='grey', lw=3, label='bias_potential'
    )
ax.set_ylim(-2, 8.5)
ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
ax.set_ylabel("potential", fontsize=20)
ax.tick_params(axis='both', labelsize=20)
ax.legend()
ax.grid()

# %%
#%% 
# S U M M A R Y  O F  M O D Y F I D  F U N C T I O N S
#---------------------------------------------------------------------
# 1. create a biased potential from D1 potential classes
#---------------------------------------------------------------------
class BiasedPotential:
    """ 
    Generate a biased potential by adding a potential_class and a bias_class.

    Parameters:
        - potential_class (class, like D1): define the potential, e.g. D1.DoubleWell potential
        - bias_class (class, like D1): define the potential, e.g. D1.Gaussian potential
    """
    def __init__(self, potential_class, bias_class):
        self._potential_class = potential_class
        self._bias_class = bias_class

    # total potential
    def potential(self, x):
        return self._potential_class.potential(x) + self._bias_class.potential(x)

    # total force
    def force(self, x, h):
        F_p = self._potential_class.force(x, h)
        F_b = self._bias_class.force(x, h)
        return F_p + F_b
        
    # total hessian
    def hessian(self, x, h):
        H_p = self._potential_class.hessian(x, h)
        H_b = self._bias_class.hessian_num(x, h)
        return H_p + H_b
#---------------------------------------------------------------------
# 2. create a pertubation potential U from D1 potential classes
#---------------------------------------------------------------------
class PertubationPotential:
    """
    Generate a pertubation potential following the sign definition 'V_t = V_s + U'. 
    Where the pertubation potential is defined as difference between 
    target and simulation potential 'U = V_t -'.

    Parameters:
        - target_potential (class, like D1): define the target potential, e.g. D1.Polynomial potential
        - simulation_potential (class, like D1): define the simulation potential, e.g. D1.DoubleWell potential
    """
    def __init__(self, target_potential, simulation_potential):
        self._target_potential = target_potential
        self._simulation_potential = simulation_potential


    # pertubation potential
    def potential(self, x):
        return self._target_potential.potential(x) - self._simulation_potential.potential(x)

    # pertubation potential
    def force(self, x, h):
        F_t = self._target_potential.force(x, h)
        F_s = self._simulation_potential.force(x, h)
        return F_t - F_s
        
    # total hessian
    def hessian(self, x, h):
        H_t = self._target_potential.hessian(x, h)
        H_s = self._simulation_potential.hessian(x, h)
        return H_t - H_s
#---------------------------------------------------------------------           
# 3. decouple potential modulation from integration 
#---------------------------------------------------------------------
def B_step(system, potential, bias = None, half_step = False):
    """
    Perform a Langevin integration B-step for a given system.

    Parameters:
    - system (object): An object representing the physical system undergoing Langevin integration.
                      It should have attributes 'v' (velocity), 'm' (mass), and 'x' (position).
    - potential (object): An object representing the potential energy landscape of the system.
                         It should have a 'force' method that calculates the force at a given position.
    - bias (object or None, optional): An object representing the bias potential or pertubation potential
                                       added to the of the system. It should have a 'force' method that 
                                       calculates the force at a given position.
    - half_step (bool, optional): If True, perform a half-step integration. Default is False, performing
                                  a full-step integration. A half-step is often used in the velocity
                                  Verlet algorithm for symplectic integration.

    Returns:
    None: The function modifies the 'v' (velocity) attribute of the provided system object in place based on the
          force calculated by the provided potential object.
    """
    
    # set time step, depending on whether a half- or full step is performed
    if half_step == True:
        dt = 0.5 * system.dt
    else:
        dt = system.dt
    if bias is not None:
        system.bias_force = bias.force(system.x, system.h)[0] 
 
    system.v = system.v + (1 / system.m) * dt * potential.force(system.x, system.h)[0] 
    
    return None 

# T O D O
#---------------------------------------------------------------------
# 4. create a reweighting scheme with scalable bias potential U*alpha 
#---------------------------------------------------------------------
class scalarD1(D1.D1):
    def __init__(
        self,
        scalar_range
    ):
        """
        Create a system class that update the delta eta for an 
        array of bias strength
        """
        super(self, scalarD1)
        
        # add new parameters
        self.bias_array = scalar_range * self.bias_force
        self.delta_eta_array = self.delta_eta
        self.logM_array = scalar_range * self.logM_force
#---------------------------------------------------------------------
# 5. create a path class that recordes k steps of wirte out path
#---------------------------------------------------------------------
class Path(D1.D1):
    def __init__(
        self,
        scalar_range
    ):
        """
        Create a system or path class that add logM over a path
        """
        super(self, Path)
        
        # add new parameters
    