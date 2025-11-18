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
    """Test the integration method."""
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


def plot_integrator_comparison(time, integrator_results, pot, potential_name):
    """Create separate plots for different integrators with clear labeling.

    Args:
        time (numpy.ndarray): Time points
        integrator_results (dict): Dictionary containing results for each
                                   integrator
        pot: Potential energy object
        potential_name (str): Name of the potential for plot title
    """
    # Set font sizes
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    TITLE_SIZE = 16

    # Set font sizes for different elements
    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=MEDIUM_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=SMALL_SIZE)
    plt.rc('figure', titlesize=TITLE_SIZE)

    # Colors for different integrators
    integrator_colors = {
        'Euler': 'red',
        'Verlet': 'blue', 
        'Leapfrog': 'green',
        'Velocity-Verlet': 'purple'
    }

    # Create figure with subplots - 2x2 grid for 4 plot types
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{potential_name} Potential - All Integrators Comparison', 
                 fontsize=TITLE_SIZE, y=0.95)

    # Plot 1: Phase space (top-left)
    ax1 = axes[0, 0]
    for name, results in integrator_results.items():
        positions = results['positions']
        velocities = results['velocities']
        momentum = velocities * results['mass']
        ax1.plot(positions, momentum, color=integrator_colors[name], 
                linewidth=1.5, label=name, alpha=0.8)
    
    ax1.set_xlabel('Position (nm)', fontweight='bold')
    ax1.set_ylabel('Momentum (amu⋅nm/ps)', fontweight='bold')
    ax1.set_title('Phase Space Trajectories', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Position trajectories (top-right)
    ax2 = axes[0, 1]
    for name, results in integrator_results.items():
        positions = results['positions']
        ax2.plot(time, positions, color=integrator_colors[name], 
                linewidth=1.5, label=name, alpha=0.8)
    
    ax2.set_xlabel('Time (ps)', fontweight='bold')
    ax2.set_ylabel('Position (nm)', fontweight='bold')
    ax2.set_title('Position vs Time', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Velocity trajectories (bottom-left)
    ax3 = axes[1, 0]
    for name, results in integrator_results.items():
        velocities = results['velocities']
        ax3.plot(time, velocities, color=integrator_colors[name], 
                linewidth=1.5, label=name, alpha=0.8)
    
    ax3.set_xlabel('Time (ps)', fontweight='bold')
    ax3.set_ylabel('Velocity (nm/ps)', fontweight='bold')
    ax3.set_title('Velocity vs Time', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Plot 4: Total energy (bottom-right)
    ax4 = axes[1, 1]
    for name, results in integrator_results.items():
        total_energy = results['total_energy']
        ax4.plot(time, total_energy, color=integrator_colors[name], 
                linewidth=1.5, label=name, alpha=0.8)
    
    ax4.set_xlabel('Time (ps)', fontweight='bold')
    ax4.set_ylabel('Total Energy (kJ/mol)', fontweight='bold')
    ax4.set_title('Total Energy vs Time', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    return fig


def plot_potential_surface(pot, potential_name, x_range=None):
    """Plot the potential energy surface."""
    if x_range is None:
        x_range = np.linspace(-3, 3, 1000)
    
    potential_values = pot.potential(x_range)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_range, potential_values, 'b-', linewidth=2)
    plt.xlabel('Position (nm)', fontweight='bold')
    plt.ylabel('Potential Energy (kJ/mol)', fontweight='bold')
    plt.title(f'{potential_name} Potential Energy Surface', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add well positions for double-well
    if hasattr(pot, 'a') and hasattr(pot, 'b'):
        well_separation = np.sqrt(pot.b)
        well_left = pot.a - well_separation
        well_right = pot.a + well_separation
        plt.axvline(well_left, color='red', linestyle='--', alpha=0.7, label=f'Left well: {well_left:.2f}')
        plt.axvline(well_right, color='red', linestyle='--', alpha=0.7, label=f'Right well: {well_right:.2f}')
        plt.axvline(pot.a, color='orange', linestyle='--', alpha=0.7, label=f'Barrier: {pot.a:.2f}')
        plt.legend()
    
    plt.tight_layout()
    return plt.gcf()


def add_diagnostic_plots(potential_configs):
    """Add diagnostic plots to understand the double-well potential behavior."""
    
    # First, let's create a diagnostic plot for the double-well potential
    print("="*80)
    print("DIAGNOSTIC: Double-Well Potential Analysis")
    print("="*80)
    
    # Test different parameter combinations
    test_params = [
        ([1.0, 0.0, 1.0], "Standard: k=1, a=0, b=1"),
        ([0.5, 0.0, 1.0], "Weaker: k=0.5, a=0, b=1"), 
        ([1.0, 0.0, 4.0], "Wider: k=1, a=0, b=4"),
        ([5.0, 0.0, 1.0], "Stronger: k=5, a=0, b=1")
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Double-Well Potential Diagnostic', fontsize=16)
    
    for idx, (params, description) in enumerate(test_params):
        ax = axes[idx//2, idx%2]
        
        # Create potential
        pot = D1.DoubleWell(params)
        k, a, b = params
        
        # Calculate well positions
        well_separation = np.sqrt(b)
        well_left = a - well_separation
        well_right = a + well_separation
        
        # Create position range
        x_range = np.linspace(-3, 3, 1000)
        potential_values = pot.potential(x_range)
        
        # Plot potential
        ax.plot(x_range, potential_values, 'b-', linewidth=2, label='Potential')
        
        # Mark wells and barrier
        ax.axvline(well_left, color='red', linestyle='--', alpha=0.7, label=f'Left well: {well_left:.2f}')
        ax.axvline(well_right, color='red', linestyle='--', alpha=0.7, label=f'Right well: {well_right:.2f}')
        ax.axvline(a, color='orange', linestyle='--', alpha=0.7, label=f'Barrier: {a:.2f}')
        
        # Calculate and show key values
        barrier_height = pot.potential(a)
        well_depth_left = pot.potential(well_left)
        well_depth_right = pot.potential(well_right)
        
        ax.set_title(f'{description}\nBarrier: {barrier_height:.2f}, Wells: {well_depth_left:.2f}')
        ax.set_xlabel('Position (nm)')
        ax.set_ylabel('Potential Energy (kJ/mol)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Set reasonable y-limits
        y_max = max(barrier_height * 1.2, 10)
        ax.set_ylim(-1, y_max)
        
        print(f"\n{description}:")
        print(f"  Well positions: {well_left:.3f}, {well_right:.3f}")
        print(f"  Barrier height: {barrier_height:.3f}")
        print(f"  Well depths: {well_depth_left:.3f}, {well_depth_right:.3f}")
    
    plt.tight_layout()
    plt.savefig('D1_plots/double_well_diagnostic.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return test_params


def run_integrator_tests(potential_configs):
    """Run tests for multiple potential configurations with enhanced diagnostics."""
    
    # First run diagnostics
    test_params = add_diagnostic_plots(potential_configs)
    
    # Simulation parameters
    dt = 0.001  # Smaller time step for double-well
    n_steps = 10000  # More steps to see behavior
    mass = 1.0  # amu

    # Create time array
    time = np.arange(n_steps + 1) * dt

    # List of integrator functions and their names
    integrators = [
        ('Euler', D1_integrator.euler_step),
        ('Verlet', D1_integrator.verlet_step),
        ('Leapfrog', D1_integrator.leapfrog_step),
        ('Velocity-Verlet', D1_integrator.velocity_verlet_step)
    ]

    # Create a summary figure to show all potentials together
    summary_fig, summary_axes = plt.subplots(len(potential_configs), 2, 
                                           figsize=(16, 6*len(potential_configs)))
    summary_fig.suptitle('Summary: All Potentials and Integrators', fontsize=16)

    for config_idx, config in enumerate(potential_configs):
        print(f"\n{'='*80}")
        print(f"TESTING {config['name'].upper()} POTENTIAL")
        print(f"{'='*80}")
        
        # Print potential parameters
        if 'debug_info' in config:
            print(config['debug_info'])
        
        # For double-well, add extra diagnostics
        if 'Double Well' in config['name']:
            pot = config['potential']
            print(f"\nDouble-well diagnostics:")
            print(f"  Parameters: k={pot.k}, a={pot.a}, b={pot.b}")
            
            # Calculate well positions
            well_separation = np.sqrt(pot.b)
            well_left = pot.a - well_separation
            well_right = pot.a + well_separation
            
            print(f"  Well positions: {well_left:.3f}, {well_right:.3f}")
            print(f"  Barrier at: {pot.a:.3f}")
            print(f"  Barrier height: {pot.potential(pot.a):.3f}")
            print(f"  Well depths: {pot.potential(well_left):.3f}, {pot.potential(well_right):.3f}")
            print(f"  Initial position: {config['initial_x']:.3f}")
            print(f"  Initial velocity: {config['initial_v']:.3f}")
            print(f"  Initial potential energy: {pot.potential(config['initial_x']):.3f}")
            print(f"  Initial force: {pot.force(config['initial_x'], 0.001)[0]:.3f}")
        
        # Dictionary to store results for each integrator
        integrator_results = {}

        # Run simulations with all integrators
        for name, integrator_func in integrators:
            # Create system and potential for each integrator
            sys = system.D1(m=mass, x=config['initial_x'], v=config['initial_v'], 
                          T=300.0, xi=1.0, dt=dt, h=0.001)
            pot = config['potential']

            print(f"\nRunning {name} integrator...")
            
            # Run simulation
            positions, velocities = run_test_md(sys, pot, integrator_func, n_steps)

            # Calculate energies
            kinetic, potential_energy, total = calculate_energies(positions, velocities, mass, pot)

            # Store results
            integrator_results[name] = {
                'positions': positions,
                'velocities': velocities,
                'kinetic_energy': kinetic,
                'potential_energy': potential_energy,
                'total_energy': total,
                'mass': mass
            }
            
            # Print energy conservation statistics
            energy_drift = abs(total[-1] - total[0])
            energy_fluctuation = np.std(total) / np.mean(total) if np.mean(total) != 0 else np.inf
            print(f"  Energy drift: {energy_drift:.6f} kJ/mol")
            print(f"  Energy fluctuation: {energy_fluctuation:.6f}")
            print(f"  Position range: [{np.min(positions):.3f}, {np.max(positions):.3f}]")
            print(f"  Final position: {positions[-1]:.3f}")

        # Create individual comparison plot for this potential
        print(f"\nCreating plots for {config['name']} potential...")
        fig = plot_integrator_comparison(time, integrator_results, pot, config['name'])

        # Save individual plot
        filename = f"D1_plots/integrator_comparison_{config['name'].lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {filename}")

        # Plot potential surface
        pot_fig = plot_potential_surface(pot, config['name'], config.get('x_range'))
        pot_filename = f"D1_plots/potential_surface_{config['name'].lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plt.savefig(pot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved potential plot: {pot_filename}")

        # Add to summary plot
        if len(potential_configs) == 1:
            summary_ax1 = summary_axes[0]
            summary_ax2 = summary_axes[1]
        else:
            summary_ax1 = summary_axes[config_idx, 0]
            summary_ax2 = summary_axes[config_idx, 1]

        # Summary plot 1: Potential surface
        x_range = config.get('x_range', np.linspace(-3, 3, 1000))
        potential_values = pot.potential(x_range)
        summary_ax1.plot(x_range, potential_values, 'b-', linewidth=2)
        summary_ax1.set_xlabel('Position (nm)')
        summary_ax1.set_ylabel('Potential Energy (kJ/mol)')
        summary_ax1.set_title(f'{config["name"]} Potential')
        summary_ax1.grid(True, alpha=0.3)

        # Add well positions for double-well
        if hasattr(pot, 'a') and hasattr(pot, 'b'):
            well_separation = np.sqrt(pot.b)
            well_left = pot.a - well_separation
            well_right = pot.a + well_separation
            summary_ax1.axvline(well_left, color='red', linestyle='--', alpha=0.7)
            summary_ax1.axvline(well_right, color='red', linestyle='--', alpha=0.7)
            summary_ax1.axvline(pot.a, color='orange', linestyle='--', alpha=0.7)

        # Summary plot 2: Phase space for all integrators
        integrator_colors = {'Euler': 'red', 'Verlet': 'blue', 'Leapfrog': 'green', 'Velocity-Verlet': 'purple'}
        for name, results in integrator_results.items():
            positions = results['positions']
            velocities = results['velocities']
            momentum = velocities * results['mass']
            summary_ax2.plot(positions, momentum, color=integrator_colors[name], 
                           linewidth=1, label=name, alpha=0.8)
        
        summary_ax2.set_xlabel('Position (nm)')
        summary_ax2.set_ylabel('Momentum (amu⋅nm/ps)')
        summary_ax2.set_title(f'{config["name"]} - Phase Space')
        summary_ax2.grid(True, alpha=0.3)
        summary_ax2.legend()

        # Show individual plots
        plt.show()

    # Save and show summary plot
    plt.figure(summary_fig.number)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('D1_plots/summary_all_potentials.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved summary plot: D1_plots/summary_all_potentials.png")
    plt.show()

    print(f"\n{'='*80}")
    print("ALL TESTS COMPLETED")
    print(f"{'='*80}")


if __name__ == "__main__":
    # Define potential configurations to test
    potential_configs = [
        {
            'name': 'Quadratic',
            'potential': D1.Quadratic([10.0, 0.0]),  # k=10.0, x0=0.0
            'initial_x': 1.0,
            'initial_v': 0.0,
            'x_range': np.linspace(-2, 2, 1000),
            'debug_info': """QUADRATIC POTENTIAL
V(x) = 0.5 * k * (x - x0)^2
Parameters: k=10.0, x0=0.0
This is a simple harmonic oscillator potential."""
        },
        {
            'name': 'Double Well (Shallow)',
            'potential': D1.DoubleWell([0.5, 0.0, 1.0]),  # k=0.5, a=0.0, b=1.0
            'initial_x': 0.9,  # Start near right well but not exactly at minimum
            'initial_v': 0.5,  # Small initial velocity
            'x_range': np.linspace(-2.5, 2.5, 1000),
            'debug_info': """DOUBLE-WELL POTENTIAL (SHALLOW)
V(x) = k * ((x - a)^2 - b)^2
Parameters: k=0.5, a=0.0, b=1.0
Well positions: x = ±√b = ±1.0
Barrier at: x = a = 0.0
Starting position: x = 0.9 (near right well)"""
        },
        {
            'name': 'Double Well (Barrier Crossing)',
            'potential': D1.DoubleWell([1.0, 0.0, 1.0]),  # k=1.0, a=0.0, b=1.0
            'initial_x': 0.0,  # Start at barrier
            'initial_v': 3.0,  # High velocity for barrier crossing
            'x_range': np.linspace(-2.5, 2.5, 1000),
            'debug_info': """DOUBLE-WELL POTENTIAL (BARRIER CROSSING)
V(x) = k * ((x - a)^2 - b)^2
Parameters: k=1.0, a=0.0, b=1.0
Well positions: x = ±√b = ±1.0
Barrier at: x = a = 0.0
Starting position: x = 0.0 (at barrier)
High velocity for barrier crossing"""
        }
    ]
    
    # Run tests for all configurations
    run_integrator_tests(potential_configs)
