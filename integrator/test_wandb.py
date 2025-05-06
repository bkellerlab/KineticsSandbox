"""
This module implements integration testing and performance analysis using Weights & Biases (wandb).

Weights & Biases (wandb) is a machine learning platform that helps track experiments, 
visualize results, and compare different configurations. In this context, it's used to:
1. Compare different numerical integrators (Euler, Verlet, Leapfrog, Velocity Verlet)
2. Analyze their performance across various potential energy functions
3. Track and visualize energy conservation, time reversal symmetry, and phase space trajectories
4. Log system performance metrics (CPU usage, memory consumption)

To use this module:
1. Install wandb: pip install wandb
2. Sign up at https://wandb.ai/ and get your API key
3. Run wandb login and enter your API key
4. Execute this script to start the parameter sweep

The sweep will test different combinations of:
- Integration methods
- Time steps
- Potential energy functions
- System parameters

Results are automatically uploaded to your wandb dashboard for visualization and analysis.
"""

import _sys
_sys.path.append("..")

import numpy as np
import wandb
from system import system
from potential import D1
from integrator import D1_integrator
import psutil


def test_euler(sys, pot, n_steps):
    """Test the Euler integration method for molecular dynamics simulation.
    
    The Euler method is the simplest numerical integration method. It updates positions
    and velocities using first-order approximations:
    x(t + dt) = x(t) + v(t) * dt
    v(t + dt) = v(t) + a(t) * dt
    
    Args:
        sys: System object containing:
            - m (float): Mass of the particle
            - x (float): Current position
            - v (float): Current velocity
            - dt (float): Time step size
            - h (float): Step size for force calculation
        pot: Potential energy object defining the force field and energy landscape
        n_steps (int): Number of integration steps to perform
        
    Returns:
        tuple: Two numpy arrays containing:
            - positions: Array of shape (n_steps + 1,) with particle positions at each time step
            - velocities: Array of shape (n_steps + 1,) with particle velocities at each time step
    """

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
    """Test the Verlet integration method for molecular dynamics simulation.
    
    The Verlet method is a second-order symplectic integrator that provides better
    energy conservation than Euler. It updates positions using:
    x(t + dt) = 2x(t) - x(t - dt) + a(t) * dt^2
    
    Note: Velocities are calculated using finite differences of positions.
    
    Args:
        sys: System object containing:
            - m (float): Mass of the particle
            - x (float): Current position
            - v (float): Current velocity
            - dt (float): Time step size
            - h (float): Step size for force calculation
        pot: Potential energy object defining the force field and energy landscape
        n_steps (int): Number of integration steps to perform
        
    Returns:
        tuple: Two numpy arrays containing:
            - positions: Array of shape (n_steps + 1,) with particle positions at each time step
            - velocities: Array of shape (n_steps,) with particle velocities at each time step
                Note: One less velocity than position due to finite difference calculation
    """

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
    """Test the Leapfrog integration method for molecular dynamics simulation.
    
    The Leapfrog method is a second-order symplectic integrator that evaluates
    velocities at half-steps and positions at full steps:
    v(t + dt/2) = v(t - dt/2) + a(t) * dt
    x(t + dt) = x(t) + v(t + dt/2) * dt
    
    This method provides excellent energy conservation and is time-reversible.
    
    Args:
        sys: System object containing:
            - m (float): Mass of the particle
            - x (float): Current position
            - v (float): Current velocity
            - v_half (float): Velocity at half time step
            - dt (float): Time step size
            - h (float): Step size for force calculation
        pot: Potential energy object defining the force field and energy landscape
        n_steps (int): Number of integration steps to perform
        
    Returns:
        tuple: Two numpy arrays containing:
            - positions: Array of shape (n_steps + 1,) with particle positions at each time step
            - velocities: Array of shape (n_steps + 1,) with particle velocities at each time step
            Note: Velocities are interpolated to full steps for output
    """

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
    """Test the Velocity Verlet integration method for molecular dynamics simulation.
    
    The Velocity Verlet method is a second-order symplectic integrator that explicitly
    evolves both positions and velocities:
    x(t + dt) = x(t) + v(t)*dt + (1/2)a(t)*dt^2
    v(t + dt) = v(t) + (1/2)[a(t) + a(t + dt)]*dt
    
    This method provides excellent energy conservation and is time-reversible while
    maintaining explicit velocity calculations.
    
    Args:
        sys: System object containing:
            - m (float): Mass of the particle
            - x (float): Current position
            - v (float): Current velocity
            - dt (float): Time step size
            - h (float): Step size for force calculation
        pot: Potential energy object defining the force field and energy landscape
        n_steps (int): Number of integration steps to perform
        
    Returns:
        tuple: Two numpy arrays containing:
            - positions: Array of shape (n_steps + 1,) with particle positions at each time step
            - velocities: Array of shape (n_steps + 1,) with particle velocities at each time step
    """

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
    """Calculate kinetic, potential, and total energy for the molecular dynamics trajectory.
    
    This function computes the energy components at each time step to analyze
    energy conservation and system behavior:
    - Kinetic Energy: K = (1/2)mv^2
    - Potential Energy: V(x) depends on the potential type
    - Total Energy: E = K + V
    
    Args:
        positions (numpy.ndarray): Array of position values for each time step
        velocities (numpy.ndarray): Array of velocity values for each time step
        mass (float): Mass of the particle in atomic mass units (amu)
        potential: Potential energy object with methods:
            - potential(x): Returns potential energy at position x
            Supported types: Quadratic, DoubleWell, and others defined in D1
        
    Returns:
        tuple: Three numpy arrays containing:
            - kinetic_energy: (1/2)mv^2 at each time step
            - potential_energy: V(x) at each time step
            - total_energy: Sum of kinetic and potential energy at each time step
            
    Note:
        Energy values are returned in units of kJ/mol
    """
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
    """Run a comprehensive Weights & Biases sweep to analyze integrator performance.
    
    This function performs the following steps:
    1. Initializes a new wandb run with configuration from the sweep
    2. Sets up a molecular dynamics system with specified parameters:
       - Mass, position, velocity
       - Temperature and friction coefficient
       - Time step and force calculation step
    3. Configures the potential energy function (various types available)
    4. Runs the selected integrator for the specified number of steps
    5. Analyzes and logs multiple metrics to wandb:
       - Energy conservation (drift and fluctuations)
       - Time reversal symmetry
       - Phase space trajectories
       - Execution time and system performance
    6. Creates visualizations including:
       - Potential energy surface
       - Position and velocity trajectories
       - Energy evolution over time
       - Phase space plots
    
    The results are automatically uploaded to your wandb dashboard where you can:
    - Compare different integrators
    - Analyze energy conservation properties
    - Evaluate computational efficiency
    - Visualize system dynamics
    
    Note:
        Requires wandb account and login. See module docstring for setup instructions.
    """
    # Initialize wandb first
    wandb.init()
    
    # Now we can safely access config to create the run name
    config = wandb.config
    run_name = f"{config.integrator}-m{config.mass}-dt{config.time_step}-{config.potential_type}"
    # Update the run name
    wandb.run.name = run_name
    wandb.run.save()
    
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
    if config.potential_type == "constant":
        pot = D1.Constant([config.d])
    elif config.potential_type == "linear":
        pot = D1.Linear([config.k_lin, config.x0])
    elif config.potential_type == "quadratic":
        pot = D1.Quadratic([config.k, config.x0])
    elif config.potential_type == "double_well":
        pot = D1.DoubleWell([config.k, config.a, config.b])
    elif config.potential_type == "polynomial":
        pot = D1.Polynomial([config.a, config.c1, config.c2, config.c3, config.c4, config.c5, config.c6])
    elif config.potential_type == "bolhuis":
        pot = D1.Bolhuis([config.a, config.b, config.c, config.k1, config.k2, config.alpha])
    #elif config.potential_type == "prinz":
    #    pot = D1.Prinz()
    elif config.potential_type == "logistic":
        pot = D1.Logistic([config.logistic_k, config.logistic_a, config.logistic_b])
    elif config.potential_type == "gaussian":
        pot = D1.Gaussian([config.gaussian_k, config.gaussian_mu, config.gaussian_sigma])
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
    mean_energy = np.mean(total_energy)
    energy_fluctuation = np.std(total_energy) / mean_energy if mean_energy != 0 else np.inf

    # Test time reversal symmetry
    # Reverse velocities and run simulation backwards
    sys_rev = system.D1(m=m, x=positions[-1], v=-velocities[-1], T=T, xi=xi, dt=dt, h=h)

    # Create time array (in ps)
    time = np.arange(0, n_steps + 1) * dt
    
    if config.integrator == "euler":
        pos_rev, vel_rev = test_euler(sys_rev, pot, n_steps)
    elif config.integrator == "verlet":
        pos_rev, vel_rev = test_verlet(sys_rev, pot, n_steps)
    elif config.integrator == "leapfrog":
        pos_rev, vel_rev = test_leapfrog(sys_rev, pot, n_steps)
    elif config.integrator == "velocity_verlet":
        pos_rev, vel_rev = test_velocity_verlet(sys_rev, pot, n_steps)
    
    # Log metrics to wandb using time instead of step index
    for i in range(0, n_steps+1):
        wandb.log({
            "time": time[i],  # Use time instead of step number
            "position": positions[i],
            "TR position": positions[i]-pos_rev[n_steps-i],
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
        # Log CPU and memory usage
        wandb.log({
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_GB": psutil.virtual_memory().used / (1024**3)
        })
    except ImportError:
        print("psutil not installed. Install with 'pip install psutil' for CPU/memory monitoring.")
    
    # Create generic potential energy surface plot (independent of trajectory)
    if config.potential_type == "quadratic":
        x_range = np.linspace(-5.0, 5.0, 1000)  # Fixed range for better visualization
    elif config.potential_type == "double_well":
        x_range = np.linspace(-2*config.a, 2*config.a, 1000)  # Show both wells
    else:
        x_range = np.linspace(-5.0, 5.0, 1000)  # Default range
        
    potential_surface = [pot.potential(x) for x in x_range]
    potential_data = [[x, v] for x, v in zip(x_range, potential_surface)]
    potential_table = wandb.Table(data=potential_data, columns=["position (nm)", "potential (kJ/mol)"])

    # Create trajectory and energy data with time
    data = [[t, positions[i], velocities[min(i, len(velocities)-1)], 
             kinetic_energy[i], potential_energy[i], total_energy[i]] 
            for i, t in enumerate(time)]
    
    table = wandb.Table(data=data, columns=["time", "position", "velocity", 
                                           "kinetic_energy", "potential_energy", "total_energy"])
    
    # Create phase space data (momentum vs position)
    momentum = [m * velocities[min(i, len(velocities)-1)] for i in range(0, n_steps+1)]
    phase_space_data = [[positions[i], momentum[i]] for i in range(0, n_steps+1)]
    phase_space_table = wandb.Table(data=phase_space_data, columns=["position (nm)", "momentum (amu·nm/ps)"])
    
    # Calculate time reversal error
    time_reversal_error = np.mean(np.abs(positions - pos_rev[::-1]))
    wandb.log({"time_reversal_error": time_reversal_error})
    
    # Create time reversal comparison data
    # Log each timestep separately to ensure data is properly captured
    for i, t in enumerate(time):
        wandb.log({
            "time_step": t,
            "forward_trajectory": positions[i],
            "reversed_trajectory": pos_rev[-(i+1)]
        })
    
    # Create the plot using custom data
    wandb.log({
        "potential_surface": wandb.plot.line(
            potential_table, 
            x="position (nm)", 
            y="potential (kJ/mol)",
            title=f"Potential Energy Surface: V(x) ({config.potential_type})"
        ),
        "trajectory_plot": wandb.plot.line(
            table, "time", "position",
            title=f"Position vs Time (ps) ({config.integrator})"),
        "TR_trajectory_plot": wandb.plot.line(
            table, "time", "TR position",
            title=f"TR Position vs Time (ps) ({config.integrator})"),
        "velocity_plot": wandb.plot.line(
            table, "time", "velocity",
            title=f"Velocity vs Time (ps) ({config.integrator})"),
        "kinetic_energy_plot": wandb.plot.line(
            table, "time", "kinetic_energy", 
            title=f"Kinetic Energy vs Time (ps) ({config.integrator})"),
        "potential_energy_plot": wandb.plot.line(
            table, "time", "potential_energy", 
            title=f"Potential Energy vs Time (ps) ({config.integrator})"),
        "total_energy_plot": wandb.plot.line(
            table, "time", "total_energy", 
            title=f"Total Energy vs Time (ps) ({config.integrator})"),
        "phase_space_plot": wandb.plot.scatter(
            phase_space_table, 
            x="position (nm)", 
            y="momentum (amu·nm/ps)", 
            title=f"Phase Space: Momentum vs Position ({config.integrator})"),
    })


def define_sweep_config():
    """Define the configuration space for the Weights & Biases parameter sweep.
    
    This function sets up a grid search over different combinations of:
    1. Integration methods:
       - Euler (first-order)
       - Verlet (second-order)
       - Leapfrog (second-order, symplectic)
       - Velocity Verlet (second-order, symplectic)
       
    2. System parameters:
       - Mass (amu)
       - Initial position (nm)
       - Initial velocity (nm/ps)
       - Temperature (K)
       - Friction coefficient (ps^-1)
       
    3. Integration parameters:
       - Time step (ps)
       - Force calculation step (nm)
       - Number of steps
       
    4. Potential energy functions:
       - Linear: V(x) = kx + x0
       - Quadratic: V(x) = (1/2)k(x - x0)^2
       - Double Well: V(x) = a(x^2 - b)^2
       - Polynomial: V(x) = sum(ci * x^i)
       And their associated parameters
    
    Returns:
        dict: Complete sweep configuration including:
            - method: Search strategy (grid, random, or Bayesian)
            - metric: Objective to optimize
            - parameters: Dictionary of parameter spaces to explore
            
    Note:
        The sweep will test all combinations of parameters in a grid search,
        so choose parameter ranges carefully to manage computational cost.
    """
    sweep_config = {
        'method': 'grid',  # or 'random', 'bayes'
        'name': 'integrator-comparison-sweep',
        'metric': {
            'name': 'energy_fluctuation',
            'goal': 'minimize'
        },
        'parameters': {
            'integrator': {
                'values': ['euler', 'leapfrog', 'verlet', 'velocity_verlet']
            },
            'mass': {
                'values': [1.0]  # Mass in atomic mass units (amu)
            },
            'initial_position': {
                'values': [1.0]  # Initial position in nm
            },
            'initial_velocity': {
                'values': [50.0]  # This won't be used as we're using thermal velocity
            },
            'temperature': {
                'values': [300.0]  # Temperature in Kelvin
            },
            'friction': {
                'values': [1.0]  # Friction coefficient in ps^-1
            },
            'time_step': {
                'values': [0.001, 0.01, 0.1]  # Time step in ps
            },
            'force_step': {
                'values': [0.001]  # Force calculation step in nm
            },
            'n_steps': {
                'values': [10000]  # Increased number of steps for better statistics
            },
            'potential_type': {
                'values': ['linear', 'quadratic', 'double_well', 'polynomial'] #['constant', 'linear', 'quadratic', 'double_well', 'polynomial', 'bolhuis', 'logistic', 'gaussian']
            },
            'd': {
                'values': [1.0]  # Constant parameter d
            },
            'k_lin': {
                'values': [10.0]  # Spring constant in kJ/(mol·nm)
            },
            'k': {
                'values': [10.0]  # Spring constant in kJ/(mol·nm²)
            },
            'x0': {
                'values': [0.0]  # Equilibrium position in nm
            },
            'a': {
                'values': [10.0]  # Double well parameter a in nm
            },
            'b': {
                'values': [1.0]  # Double well parameter b in nm
            },
            'c1': {
                'values': [1.0]  # Polynomial parameter c1 in kJ/(mol·nm)
            },
            'c2': {
                'values': [1.0]  # Polynomial parameter c2 in kJ/(mol·nm²)
            },
            'c3': {
                'values': [1.0]  # Polynomial parameter c3 in kJ/(mol·nm³)
            },
            'c4': {
                'values': [1.0]  # Polynomial parameter c4 in kJ/(mol·nm⁴)
            },
            'c5': {
                'values': [1.0]  # Polynomial parameter c5 in kJ/(mol·nm⁵)
            },
            'c6': {
                'values': [1.0]  # Polynomial parameter c6 in kJ/(mol·nm⁶)
            },
            'c': {
                'values': [1.0]  # Bolhuis parameter c
            },
            'k1': {
                'values': [1.0]  # Bolhuis parameter k1 in kJ/(mol·nm²)
            },
            'k2': {
                'values': [1.0]  # Bolhuis parameter k2 in kJ/(mol·nm)
            },
            'alpha': {
                'values': [1.0]  # Bolhuis parameter alpha in nm
            },
            'logistic_k': {
                'values': [1.0]  # Logistic parameter k in kJ/(mol)
            },
            'logistic_a': {
                'values': [1.0]  # Logistic parameter a in nm
            },
            'logistic_b': {
                'values': [1.0]  # Logistic parameter b in nm
            },
            'gaussian_k': {
                'values': [1.0]  # Gaussian parameter k
            },
            'gaussian_mu': {
                'values': [1.0]  # Gaussian parameter mu
            },
            'gaussian_sigma': {
                'values': [1.0]  # Gaussian parameter sigma
            },
        }
    }
    return sweep_config


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define sweep configuration
    sweep_config = define_sweep_config()
    
    # Initialize sweep with a descriptive name
    sweep_id = wandb.sweep(
        sweep_config, 
        project="integrator-comparison",
        entity=wandb.api.default_entity
    )
    
    # Run the sweep
    wandb.agent(sweep_id, function=run_wandb_sweep)
