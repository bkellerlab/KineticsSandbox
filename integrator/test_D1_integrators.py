"""
This module provides testing and visualization for different molecular dynamics integrators
without requiring Weights & Biases (wandb).
"""

import numpy as np
import matplotlib.pyplot as plt
import sys as _sys
_sys.path.append("..")

from system import system
from potential import D1
from integrator import D1_integrator


def run_test_md(sys, pot, integrator, n_steps):
    """Test the Euler integration method."""
    positions = np.zeros(n_steps + 1)
    velocities = np.zeros(n_steps + 1)

    # Store initial state
    positions[0] = sys.x
    velocities[0] = sys.v

    # Integration loop
    for i in range(n_steps):
        integrator(sys, pot)
        # Store current state
        positions[i+1] = sys.x
        velocities[i+1] = sys.v

    return positions, velocities


def calculate_energies(positions, velocities, mass, potential):
    """Calculate kinetic, potential, and total energy for the trajectory."""

    kinetic_energy = 0.5 * mass * velocities**2
    potential_energy = potential.potential(positions)
    total_energy = kinetic_energy + potential_energy

    return kinetic_energy, potential_energy, total_energy


def plot_integrator_comparison(time, integrator_results, pot):
    """Create separate plots for different integrators.

    Args:
        time (numpy.ndarray): Time points
        integrator_results (dict): Dictionary containing results for each
                                   integrator
        pot: Potential energy object
    """
    # Set font sizes
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    # BIGGER_SIZE = 14
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
    plt.suptitle('Comparison of Molecular Dynamics Integrators',
                 fontsize=TITLE_SIZE, y=0.95)

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
    dt = 0.01  # Time step (ps)
    n_steps = 5000  # Number of steps
    mass = 1.0  # amu

    # Create time array
    time = np.arange(n_steps + 1) * dt

    # Dictionary to store results for each integrator
    integrator_results = {}

    # List of integrator functions and their names
    integrators = [
        ('Euler', D1_integrator.euler_step),
        ('Verlet', D1_integrator.verlet_step),
        ('Leapfrog', D1_integrator.leapfrog_step),
        ('Velocity-Verlet', D1_integrator.velocity_verlet_step)
    ]

    # Run simulations with all integrators
    for name, integrator_func in integrators:
        # Create system and potential for each integrator
        sys = system.D1(m=mass, x=1.0, v=0.0, T=300.0, xi=1.0, dt=dt, h=0.001)
        pot = D1.Quadratic([10.0, 0.0])  # k=10.0, x0=0.0

        # Run simulation
        positions, velocities = run_test_md(sys, pot, integrator_func, n_steps)

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
