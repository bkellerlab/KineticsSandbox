import sys
sys.path.append("..")

import numpy as np
import wandb
from system import system
from potential import D1
from integrator import D1_integrator

# Non-zero initial position with smaller mass and larger timestep
np.random.seed(42)

# System parameters
m = 100.0
x = 0.0
v = 0.0 
T = 300.0
xi = 1.0
dt = 0.001
h = 0.001
n_steps = 1000

sys = system.D1(m=m, x=x, v=v, T=T, xi=xi, dt=dt, h=h)
pot = D1.Quadratic([1.0, 0.0])


def test_euler(sys, pot, n_steps):

    positions = np.zeros(n_steps + 1)
    velocities = np.zeros(n_steps + 1)

    # Store initial state
    positions[0] = sys.x
    velocities[0] = sys.v

    # Integration loop
    for i in range(n_steps):
        D1_integrator.euler_step(sys, pot)
        # Store current state
        positions[i+1] = sys.x
        velocities[i+1] = sys.v

    return positions, velocities


def test_verlet(sys, pot, n_steps):

    positions = np.zeros(n_steps + 1)
    velocities = np.zeros(n_steps)  # One less velocity than positions

    # Store initial state
    positions[0] = sys.x

    # Integration loop
    for i in range(n_steps):
        D1_integrator.verlet_step(sys, pot)
        # Store current state
        positions[i+1] = sys.x
        if i > 0:  # Store velocity from previous step
            velocities[i-1] = sys.v

    # Calculate final velocity
    velocities[-1] = (positions[-1] - positions[-2]) / sys.dt 

    return positions, velocities


def test_leapfrog(sys, pot, n_steps):

    positions = np.zeros(n_steps + 1)
    velocities = np.zeros(n_steps + 1)
    velocity_halfsteps = np.zeros(n_steps + 2)  # +2 for initial and final half steps

    # Store initial state
    positions[0] = sys.x
    velocities[0] = sys.v

    # Calculate initial force
    force = pot.force(positions[0], sys.h)[0]

    # Initialize velocity half steps exactly as in leapfrog()
    velocity_halfsteps[0] = sys.v - (force/sys.m) * (sys.dt/2)
    velocity_halfsteps[1] = sys.v + (force/sys.m) * (sys.dt/2)

    # Set the system's v_half to match
    sys.v_half = velocity_halfsteps[1]

    # Integration loop
    for i in range(n_steps):
        # Store the current position
        x_current = sys.x
        
        # Calculate force at current position
        force = pot.force(x_current, sys.h)[0]
        
        # Update velocity at half step (t+1)
        velocity_halfsteps[i+1] = velocity_halfsteps[i] + (force/sys.m) * sys.dt
        
        # Update position using half-step velocity
        sys.x = x_current + velocity_halfsteps[i+1] * sys.dt
        
        # Calculate velocity at full step (for output)
        sys.v = (velocity_halfsteps[i+1] + velocity_halfsteps[i]) / 2
        
        # Store current state
        positions[i+1] = sys.x
        velocities[i+1] = sys.v

    return positions, velocities
    
    
def test_velocity_verlet(sys, pot, n_steps):

    positions = np.zeros(n_steps + 1)
    velocities = np.zeros(n_steps + 1)

    # Store initial state
    positions[0] = sys.x
    velocities[0] = sys.v

    # Integration loop
    for i in range(n_steps):
        D1_integrator.velocity_verlet_step(sys, pot)
        
        # Store current state
        positions[i+1] = sys.x
        velocities[i+1] = sys.v

    return positions, velocities


def calculate_energies(positions, velocities, mass, potential):
    """Calculate kinetic, potential, and total energy for the trajectory"""
    n_steps = len(positions) - 1
    
    # Initialize energy arrays
    kinetic_energy = np.zeros(n_steps + 1)
    potential_energy = np.zeros(n_steps + 1)
    total_energy = np.zeros(n_steps + 1)
    
    # Calculate energies
    for i in range(n_steps + 1):
        # Kinetic energy: 0.5 * m * v^2
        if i < len(velocities):
            kinetic_energy[i] = 0.5 * mass * velocities[i]**2
        
        # Potential energy - calculate based on potential type
        if isinstance(potential, D1.Quadratic):
            # For quadratic potential: 0.5 * k * (x - x0)^2
            # Use the potential's own method to calculate energy
            potential_energy[i] = potential.potential(positions[i])
        elif isinstance(potential, D1.DoubleWell):
            # For double well potential: a*(x^2 - b)^2
            potential_energy[i] = potential.potential(positions[i])
        else:
            # Default to zero if potential type is unknown
            potential_energy[i] = 0.0
        
        # Total energy
        total_energy[i] = kinetic_energy[i] + potential_energy[i]
    
    return kinetic_energy, potential_energy, total_energy


def run_wandb_sweep():
    # Initialize wandb
    wandb.init()
    
    # Get parameters from wandb config
    config = wandb.config
    
    # System parameters
    m = config.mass
    x = config.initial_position
    v = config.initial_velocity
    T = config.temperature
    xi = config.friction
    dt = config.time_step
    h = config.force_step
    n_steps = config.n_steps
    
    # Create system and potential
    sys = system.D1(m=m, x=x, v=v, T=T, xi=xi, dt=dt, h=h)
    
    # Select potential based on config
    if config.potential_type == "quadratic":
        pot = D1.Quadratic([config.k, config.x0])
    elif config.potential_type == "double_well":
        pot = D1.DoubleWell([config.a, config.b, 0.0])  # Adding a third parameter with default value 0.0
    else:
        pot = D1.Quadratic([1.0, 0.0])  # Default
    
    # Track execution time
    import time
    start_time = time.time()
    
    # Select integrator based on config
    if config.integrator == "euler":
        positions, velocities = test_euler(sys, pot, n_steps)
    elif config.integrator == "verlet":
        positions, velocities = test_verlet(sys, pot, n_steps)
    elif config.integrator == "leapfrog":
        positions, velocities = test_leapfrog(sys, pot, n_steps)
    elif config.integrator == "velocity_verlet":
        positions, velocities = test_velocity_verlet(sys, pot, n_steps)
    
    # Calculate execution time
    execution_time = time.time() - start_time
    
    # Calculate energies
    kinetic_energy, potential_energy, total_energy = calculate_energies(
        positions, velocities, m, pot
    )
    
    # Calculate energy conservation metrics
    energy_drift = np.abs(total_energy[-1] - total_energy[0])
    energy_fluctuation = np.std(total_energy) / np.mean(total_energy)
    
    # Log metrics to wandb
    for i in range(0, n_steps+1, max(1, n_steps//100)):  # Log ~100 points
        wandb.log({
            "step": i,
            "position": positions[i],
            "velocity": velocities[min(i, len(velocities)-1)],
            "kinetic_energy": kinetic_energy[i],
            "potential_energy": potential_energy[i],
            "total_energy": total_energy[i]
        })
    
    # Log summary metrics including performance metrics
    wandb.log({
        "energy_drift": energy_drift,
        "energy_fluctuation": energy_fluctuation,
        "final_position": positions[-1],
        "final_velocity": velocities[-1],
        "execution_time": execution_time,
    })
    
    # For more detailed system monitoring, you can use psutil
    try:
        import psutil
        # Log CPU and memory usage
        wandb.log({
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_GB": psutil.virtual_memory().used / (1024**3)
        })
    except ImportError:
        print("psutil not installed. Install with 'pip install psutil' for CPU/memory monitoring.")
    
    # Create and log trajectory plot
    data = [[i, positions[i], velocities[min(i, len(velocities)-1)], 
             kinetic_energy[i], potential_energy[i], total_energy[i]] 
            for i in range(0, n_steps+1, max(1, n_steps//100))]
    
    table = wandb.Table(data=data, columns=["step", "position", "velocity", 
                                           "kinetic_energy", "potential_energy", "total_energy"])
    
    wandb.log({
        "trajectory_plot": wandb.plot.line(
            table, "step", "position", title="Position Trajectory"),
        "energy_plot": wandb.plot.line(
            table, "step", ["kinetic_energy", "potential_energy", "total_energy"], 
            title="Energy Components")
    })


def define_sweep_config():
    """Define the sweep configuration"""
    sweep_config = {
        'method': 'grid',  # or 'random', 'bayes'
        'metric': {
            'name': 'energy_fluctuation',
            'goal': 'minimize'
        },
        'parameters': {
            'integrator': {
                'values': ['euler', 'verlet', 'leapfrog', 'velocity_verlet']
            },
            'mass': {
                'values': [1.0, 10.0, 100.0]
            },
            'initial_position': {
                'values': [0.0, 1.0]
            },
            'initial_velocity': {
                'values': [0.0]
            },
            'temperature': {
                'values': [300.0]
            },
            'friction': {
                'values': [1.0]
            },
            'time_step': {
                'values': [0.001, 0.01, 0.1]
            },
            'force_step': {
                'values': [0.001]
            },
            'n_steps': {
                'values': [1000]
            },
            'potential_type': {
                'values': ['quadratic', 'double_well']
            },
            'k': {
                'values': [1.0]
            },
            'x0': {
                'values': [0.0]
            },
            'a': {
                'values': [1.0]
            },
            'b': {
                'values': [0.0]
            }
        }
    }
    return sweep_config


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define sweep configuration
    sweep_config = define_sweep_config()
    
    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project="integrator-comparison")
    
    # Run the sweep
    wandb.agent(sweep_id, function=run_wandb_sweep)