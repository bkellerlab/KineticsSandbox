"""
This module provides testing and visualization for different molecular dynamics integrators
without requiring Weights & Biases (wandb).
"""

import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
from system import system
from potential import D1
from integrator import D1_integrator


def test_euler(sys, pot, n_steps):
    """Test the Euler integration method."""
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
    """Test the Verlet integration method."""
    positions = np.zeros(n_steps + 1)
    velocities = np.zeros(n_steps + 1)  # Changed to n_steps + 1

    # Store initial state
    positions[0] = sys.x
    # Calculate initial velocity using half-step
    force = pot.force(sys.x, sys.h)[0]
    velocities[0] = sys.v  # Store initial velocity

    # Integration loop
    for i in range(n_steps):
        D1_integrator.verlet_step(sys, pot)
        # Store current state
        positions[i+1] = sys.x
        if i < n_steps - 1:  # Calculate velocity for all but last step
            velocities[i+1] = (positions[i+2] - positions[i]) / (2 * sys.dt)

    # Calculate final velocity using backward difference
    velocities[-1] = (positions[-1] - positions[-2]) / sys.dt

    return positions, velocities


def test_leapfrog(sys, pot, n_steps):
    """Test the Leapfrog integration method."""
    positions = np.zeros(n_steps + 1)
    velocities = np.zeros(n_steps + 1)
    velocity_halfsteps = np.zeros(n_steps + 2)  # +2 for initial and final half steps

    # Store initial state
    positions[0] = sys.x
    velocities[0] = sys.v

    # Calculate initial force
    force = pot.force(positions[0], sys.h)[0]

    # Initialize velocity half steps
    velocity_halfsteps[0] = sys.v - (force/sys.m) * (sys.dt/2)
    velocity_halfsteps[1] = sys.v + (force/sys.m) * (sys.dt/2)

    # Set the system's v_half to match
    sys.v_half = velocity_halfsteps[1]

    # Integration loop
    for i in range(n_steps):
        x_current = sys.x
        force = pot.force(x_current, sys.h)[0]
        velocity_halfsteps[i+1] = velocity_halfsteps[i] + (force/sys.m) * sys.dt
        sys.x = x_current + velocity_halfsteps[i+1] * sys.dt
        sys.v = (velocity_halfsteps[i+1] + velocity_halfsteps[i]) / 2
        positions[i+1] = sys.x
        velocities[i+1] = sys.v

    return positions, velocities


def test_velocity_verlet(sys, pot, n_steps):
    """Test the Velocity Verlet integration method."""
    positions = np.zeros(n_steps + 1)
    velocities = np.zeros(n_steps + 1)

    # Store initial state
    positions[0] = sys.x
    velocities[0] = sys.v

    # Integration loop
    for i in range(n_steps):
        D1_integrator.velocity_verlet_step(sys, pot)
        positions[i+1] = sys.x
        velocities[i+1] = sys.v

    return positions, velocities


def calculate_energies(positions, velocities, mass, potential):
    """Calculate kinetic, potential, and total energy for the trajectory."""
    n_steps = len(positions) - 1
    
    kinetic_energy = np.zeros(n_steps + 1)
    potential_energy = np.zeros(n_steps + 1)
    total_energy = np.zeros(n_steps + 1)
    
    for i in range(n_steps + 1):
        # Kinetic energy
        if i < len(velocities):
            kinetic_energy[i] = 0.5 * mass * velocities[i]**2
        
        # Potential energy
        potential_energy[i] = potential.potential(positions[i])
        
        # Total energy
        total_energy[i] = kinetic_energy[i] + potential_energy[i]
    
    return kinetic_energy, potential_energy, total_energy


def plot_results(time, positions, velocities, kinetic, potential, total, integrator_name, pot):
    """Create plots for the simulation results."""
    # Set font sizes
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14
    TITLE_SIZE = 16

    # Set font sizes for different elements
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=TITLE_SIZE)   # fontsize of the figure title

    # Create figure with subplots in a 2x2 grid with more space
    fig = plt.figure(figsize=(15, 12))
    plt.suptitle(f'Molecular Dynamics Results: {integrator_name}', fontsize=TITLE_SIZE, y=0.95)
    
    # Calculate momentum (p = mv)
    momentum = velocities * sys.m
    
    # Phase space plot with density
    ax1 = plt.subplot(2, 2, 1)
    # Create 2D histogram for density plot
    H, xedges, yedges = np.histogram2d(positions, momentum, bins=50)
    # Create a scatter plot with density-based coloring
    scatter = ax1.scatter(positions, momentum, c=np.arange(len(positions)), 
                         cmap='Reds', alpha=0.1, s=1)
    ax1.plot(positions, momentum, 'r-', alpha=0.3, linewidth=0.5)
    ax1.set_xlabel('Position (nm)', fontweight='bold')
    ax1.set_ylabel('Momentum (amu⋅nm/ps)', fontweight='bold')
    ax1.set_title('Phase Space Trajectory', pad=10, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Position trajectory plot
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(time, positions, 'k-', linewidth=1)
    ax2.set_xlabel('Time (ps)', fontweight='bold')
    ax2.set_ylabel('Position (nm)', fontweight='bold')
    ax2.set_title('Position vs Time', pad=10, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Velocity trajectory plot
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(time, velocities, 'k-', linewidth=1)
    ax3.set_xlabel('Time (ps)', fontweight='bold')
    ax3.set_ylabel('Velocity (nm/ps)', fontweight='bold')
    ax3.set_title('Velocity vs Time', pad=10, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Total energy plot
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(time, total, 'k-', linewidth=1)
    ax4.set_xlabel('Time (ps)', fontweight='bold')
    ax4.set_ylabel('Total Energy (kJ/mol)', fontweight='bold')
    ax4.set_title('Total Energy vs Time', pad=10, fontweight='bold')
    # Set y-axis limits to show small fluctuations
    mean_energy = np.mean(total)
    energy_range = np.max(total) - np.min(total)
    ax4.set_ylim(mean_energy - energy_range*2, mean_energy + energy_range*2)
    ax4.grid(True, alpha=0.3)
    
    # Adjust layout to prevent overlapping
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    return fig


def plot_integrator_comparison(time, integrator_results, pot):
    """Create separate plots for different integrators.
    
    Args:
        time (numpy.ndarray): Time points
        integrator_results (dict): Dictionary containing results for each integrator
        pot: Potential energy object
    """
    # Set font sizes
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14
    TITLE_SIZE = 16

    # Set font sizes for different elements
    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=MEDIUM_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=SMALL_SIZE)
    plt.rc('figure', titlesize=TITLE_SIZE)

    # Colors for different plots
    plot_colors = {
        'phase': 'blue',
        'position': 'red',
        'velocity': 'green',
        'energy': 'purple'
    }

    # Create figure with subplots for each integrator
    fig = plt.figure(figsize=(20, 16))
    plt.suptitle('Comparison of Molecular Dynamics Integrators', fontsize=TITLE_SIZE, y=0.95)
    
    # Plot for each integrator
    for idx, (name, results) in enumerate(integrator_results.items(), 1):
        positions = results['positions']
        velocities = results['velocities']
        momentum = velocities * results['mass']
        
        # Create subplot grid for this integrator
        # Phase space plot
        ax1 = plt.subplot(4, 4, idx)
        ax1.plot(positions, momentum, color=plot_colors['phase'], linewidth=1)
        ax1.set_xlabel('Position (nm)', fontweight='bold')
        ax1.set_ylabel('Momentum (amu⋅nm/ps)', fontweight='bold')
        ax1.set_title(f'{name}\nPhase Space', pad=10, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Position trajectory plot
        ax2 = plt.subplot(4, 4, idx + 4)
        ax2.plot(time, positions, color=plot_colors['position'], linewidth=1)
        ax2.set_xlabel('Time (ps)', fontweight='bold')
        ax2.set_ylabel('Position (nm)', fontweight='bold')
        ax2.set_title('Position vs Time', pad=10, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Velocity trajectory plot
        ax3 = plt.subplot(4, 4, idx + 8)
        ax3.plot(time, velocities, color=plot_colors['velocity'], linewidth=1)
        ax3.set_xlabel('Time (ps)', fontweight='bold')
        ax3.set_ylabel('Velocity (nm/ps)', fontweight='bold')
        ax3.set_title('Velocity vs Time', pad=10, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Total energy plot
        ax4 = plt.subplot(4, 4, idx + 12)
        ax4.plot(time, results['total_energy'], color=plot_colors['energy'], linewidth=1)
        ax4.set_xlabel('Time (ps)', fontweight='bold')
        ax4.set_ylabel('Total Energy (kJ/mol)', fontweight='bold')
        ax4.set_title('Total Energy vs Time', pad=10, fontweight='bold')
        # Set y-axis limits to show small fluctuations
        mean_energy = np.mean(results['total_energy'])
        energy_range = np.max(results['total_energy']) - np.min(results['total_energy'])
        ax4.set_ylim(mean_energy - energy_range*2, mean_energy + energy_range*2)
        ax4.grid(True, alpha=0.3)
    
    # Adjust layout to prevent overlapping
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    return fig


if __name__ == "__main__":
    # Simulation parameters
    dt = 0.001  # Time step (ps)
    n_steps = 10000  # Number of steps
    mass = 1.0  # amu
    
    # Create time array
    time = np.arange(n_steps + 1) * dt
    
    # Dictionary to store results for each integrator
    integrator_results = {}
    
    # List of integrator functions and their names
    integrators = [
        ('Euler', test_euler),
        ('Verlet', test_verlet),
        ('Leapfrog', test_leapfrog),
        ('Velocity Verlet', test_velocity_verlet)
    ]
    
    # Run simulations with all integrators
    for name, integrator_func in integrators:
        # Create system and potential for each integrator
        sys = system.D1(m=mass, x=1.0, v=0.0, T=300.0, xi=1.0, dt=dt, h=0.001)
        pot = D1.Quadratic([10.0, 0.0])  # k=10.0, x0=0.0
        
        # Run simulation
        positions, velocities = integrator_func(sys, pot, n_steps)
        
        # Calculate energies
        kinetic, potential, total = calculate_energies(positions, velocities, mass, pot)
        
        # Store results
        integrator_results[name] = {
            'positions': positions,
            'velocities': velocities,
            'kinetic_energy': kinetic,
            'potential_energy': potential,
            'total_energy': total,
            'mass': mass
        }
    
    # Create and show comparison plots
    fig = plot_integrator_comparison(time, integrator_results, pot)
    
    # Save plot with higher resolution
    plt.savefig('integrator_comparison.png', dpi=300, bbox_inches='tight')
    
    # Show plot
    plt.show()
