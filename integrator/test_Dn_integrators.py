"""
This module provides testing and visualization for different N-dimensional molecular dynamics integrators
without requiring Weights & Biases (wandb).
"""

import numpy as np
import matplotlib.pyplot as plt
import sys as _sys
_sys.path.append("..")

from system import system
from potential import Dn
from integrator import Dn_integrator


def run_test_md_nd(sys, pot, integrator, n_steps):
    """Test the N-dimensional integration method."""
    n_dims = len(sys.x)
    positions = np.zeros((n_steps + 1, n_dims))
    velocities = np.zeros((n_steps + 1, n_dims))

    # Store initial state
    positions[0] = sys.x.copy()
    velocities[0] = sys.v.copy()

    # Integration loop
    for i in range(n_steps):
        integrator(sys, pot)
        # Store current state
        positions[i+1] = sys.x.copy()
        velocities[i+1] = sys.v.copy()

    return positions, velocities


def calculate_energies_nd(positions, velocities, mass, potential):
    """Calculate kinetic, potential, and total energy for the N-D trajectory."""
    n_steps = positions.shape[0]
    
    kinetic_energy = np.zeros(n_steps)
    potential_energy = np.zeros(n_steps)
    total_energy = np.zeros(n_steps)
    
    for i in range(n_steps):
        # Kinetic energy: 0.5 * m * |v|^2
        kinetic_energy[i] = 0.5 * mass * np.sum(velocities[i]**2)
        
        # Potential energy
        potential_energy[i] = potential.potential(positions[i])
        
        # Total energy
        total_energy[i] = kinetic_energy[i] + potential_energy[i]

    return kinetic_energy, potential_energy, total_energy


def plot_integrator_comparison_nd(time, integrator_results, pot, potential_name, n_dims):
    """Create comparison plots for N-dimensional integrators.

    Args:
        time (numpy.ndarray): Time points
        integrator_results (dict): Dictionary containing results for each integrator
        pot: Potential energy object
        potential_name (str): Name of the potential for plot title
        n_dims (int): Number of dimensions
    """
    # Set font sizes
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    TITLE_SIZE = 16

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

    if n_dims == 2:
        # For 2D: Create 2x3 grid
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{potential_name} Potential (2D) - All Integrators Comparison', 
                     fontsize=TITLE_SIZE, y=0.95)

        # Plot 1: 2D trajectory (top-left)
        ax1 = axes[0, 0]
        for name, results in integrator_results.items():
            positions = results['positions']
            ax1.plot(positions[:, 0], positions[:, 1], color=integrator_colors[name], 
                    linewidth=1.5, label=name, alpha=0.8)
            # Mark starting point
            ax1.plot(positions[0, 0], positions[0, 1], 'o', color=integrator_colors[name], 
                    markersize=8, alpha=0.8)
        
        ax1.set_xlabel('X Position (nm)', fontweight='bold')
        ax1.set_ylabel('Y Position (nm)', fontweight='bold')
        ax1.set_title('2D Trajectory', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_aspect('equal')

        # Plot 2: X position vs time (top-middle)
        ax2 = axes[0, 1]
        for name, results in integrator_results.items():
            positions = results['positions']
            ax2.plot(time, positions[:, 0], color=integrator_colors[name], 
                    linewidth=1.5, label=name, alpha=0.8)
        
        ax2.set_xlabel('Time (ps)', fontweight='bold')
        ax2.set_ylabel('X Position (nm)', fontweight='bold')
        ax2.set_title('X Position vs Time', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Plot 3: Y position vs time (top-right)
        ax3 = axes[0, 2]
        for name, results in integrator_results.items():
            positions = results['positions']
            ax3.plot(time, positions[:, 1], color=integrator_colors[name], 
                    linewidth=1.5, label=name, alpha=0.8)
        
        ax3.set_xlabel('Time (ps)', fontweight='bold')
        ax3.set_ylabel('Y Position (nm)', fontweight='bold')
        ax3.set_title('Y Position vs Time', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # Plot 4: Speed vs time (bottom-left)
        ax4 = axes[1, 0]
        for name, results in integrator_results.items():
            velocities = results['velocities']
            speed = np.sqrt(np.sum(velocities**2, axis=1))
            ax4.plot(time, speed, color=integrator_colors[name], 
                    linewidth=1.5, label=name, alpha=0.8)
        
        ax4.set_xlabel('Time (ps)', fontweight='bold')
        ax4.set_ylabel('Speed (nm/ps)', fontweight='bold')
        ax4.set_title('Speed vs Time', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        # Plot 5: Total energy (bottom-middle)
        ax5 = axes[1, 1]
        for name, results in integrator_results.items():
            total_energy = results['total_energy']
            ax5.plot(time, total_energy, color=integrator_colors[name], 
                    linewidth=1.5, label=name, alpha=0.8)
        
        ax5.set_xlabel('Time (ps)', fontweight='bold')
        ax5.set_ylabel('Total Energy (kJ/mol)', fontweight='bold')
        ax5.set_title('Total Energy vs Time', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.legend()

        # Plot 6: Phase space (bottom-right) - X vs Vx
        ax6 = axes[1, 2]
        for name, results in integrator_results.items():
            positions = results['positions']
            velocities = results['velocities']
            ax6.plot(positions[:, 0], velocities[:, 0], color=integrator_colors[name], 
                    linewidth=1.5, label=name, alpha=0.8)
        
        ax6.set_xlabel('X Position (nm)', fontweight='bold')
        ax6.set_ylabel('X Velocity (nm/ps)', fontweight='bold')
        ax6.set_title('Phase Space (X)', fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.legend()

    else:
        # For higher dimensions: Create 2x2 grid with summary plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{potential_name} Potential ({n_dims}D) - All Integrators Comparison', 
                     fontsize=TITLE_SIZE, y=0.95)

        # Plot 1: First dimension position vs time (top-left)
        ax1 = axes[0, 0]
        for name, results in integrator_results.items():
            positions = results['positions']
            ax1.plot(time, positions[:, 0], color=integrator_colors[name], 
                    linewidth=1.5, label=name, alpha=0.8)
        
        ax1.set_xlabel('Time (ps)', fontweight='bold')
        ax1.set_ylabel('X₁ Position (nm)', fontweight='bold')
        ax1.set_title('First Dimension vs Time', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot 2: Distance from origin vs time (top-right)
        ax2 = axes[0, 1]
        for name, results in integrator_results.items():
            positions = results['positions']
            distance = np.sqrt(np.sum(positions**2, axis=1))
            ax2.plot(time, distance, color=integrator_colors[name], 
                    linewidth=1.5, label=name, alpha=0.8)
        
        ax2.set_xlabel('Time (ps)', fontweight='bold')
        ax2.set_ylabel('Distance from Origin (nm)', fontweight='bold')
        ax2.set_title('Radial Distance vs Time', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Plot 3: Speed vs time (bottom-left)
        ax3 = axes[1, 0]
        for name, results in integrator_results.items():
            velocities = results['velocities']
            speed = np.sqrt(np.sum(velocities**2, axis=1))
            ax3.plot(time, speed, color=integrator_colors[name], 
                    linewidth=1.5, label=name, alpha=0.8)
        
        ax3.set_xlabel('Time (ps)', fontweight='bold')
        ax3.set_ylabel('Speed (nm/ps)', fontweight='bold')
        ax3.set_title('Speed vs Time', fontweight='bold')
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

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    return fig


def plot_potential_surface_nd(pot, potential_name, n_dims, x_range=None):
    """Plot the potential energy surface for N-D systems."""
    if x_range is None:
        x_range = np.linspace(-3, 3, 100)
    
    if n_dims == 2:
        # Create 2D contour plot
        X, Y = np.meshgrid(x_range, x_range)
        Z = np.zeros_like(X)
        
        for i in range(len(x_range)):
            for j in range(len(x_range)):
                pos = np.array([X[i, j], Y[i, j]])
                Z[i, j] = pot.potential(pos)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Contour plot
        contour = ax1.contour(X, Y, Z, levels=20)
        ax1.clabel(contour, inline=True, fontsize=8)
        ax1.set_xlabel('X Position (nm)', fontweight='bold')
        ax1.set_ylabel('Y Position (nm)', fontweight='bold')
        ax1.set_title(f'{potential_name} Potential - Contour Plot', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # 3D surface plot
        ax2 = fig.add_subplot(122, projection='3d')
        surf = ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax2.set_xlabel('X Position (nm)')
        ax2.set_ylabel('Y Position (nm)')
        ax2.set_zlabel('Potential Energy (kJ/mol)')
        ax2.set_title(f'{potential_name} Potential - 3D Surface')
        fig.colorbar(surf, ax=ax2, shrink=0.5)
        
    else:
        # For higher dimensions, show 1D slices
        fig, axes = plt.subplots(1, min(n_dims, 3), figsize=(5*min(n_dims, 3), 5))
        if n_dims == 1:
            axes = [axes]
        
        for dim in range(min(n_dims, 3)):
            ax = axes[dim] if n_dims > 1 else axes[0]
            
            # Create 1D slice along this dimension
            potential_values = []
            for x_val in x_range:
                pos = np.zeros(n_dims)
                pos[dim] = x_val
                potential_values.append(pot.potential(pos))
            
            ax.plot(x_range, potential_values, 'b-', linewidth=2)
            ax.set_xlabel(f'X_{dim+1} Position (nm)', fontweight='bold')
            ax.set_ylabel('Potential Energy (kJ/mol)', fontweight='bold')
            ax.set_title(f'{potential_name} - Slice along X_{dim+1}', fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def run_integrator_tests_nd(potential_configs):
    """Run tests for multiple N-dimensional potential configurations."""
    
    # Simulation parameters
    dt = 0.001  # Time step (ps)
    n_steps = 5000  # Number of steps
    mass = 1.0  # amu

    # Create time array
    time = np.arange(n_steps + 1) * dt

    # List of integrator functions and their names
    integrators = [
        ('Euler', Dn_integrator.euler_step),
        ('Verlet', Dn_integrator.verlet_step),
        ('Leapfrog', Dn_integrator.leapfrog_step),
        ('Velocity-Verlet', Dn_integrator.velocity_verlet_step)
    ]

    for config in potential_configs:
        print(f"\n{'='*80}")
        print(f"TESTING {config['name'].upper()} POTENTIAL ({config['n_dims']}D)")
        print(f"{'='*80}")
        
        # Print potential parameters
        if 'debug_info' in config:
            print(config['debug_info'])
        
        # Dictionary to store results for each integrator
        integrator_results = {}

        # Run simulations with all integrators
        for name, integrator_func in integrators:
            # Create system for N-D
            sys = system.Dn(m=mass, x=config['initial_x'].copy(), v=config['initial_v'].copy(), 
                          T=300.0, xi=1.0, dt=dt, h=0.001)
            pot = config['potential']

            print(f"\nRunning {name} integrator...")
            
            # Run simulation
            positions, velocities = run_test_md_nd(sys, pot, integrator_func, n_steps)

            # Calculate energies
            kinetic, potential_energy, total = calculate_energies_nd(positions, velocities, mass, pot)

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
            
            # Print position statistics
            final_distance = np.sqrt(np.sum(positions[-1]**2))
            max_distance = np.max(np.sqrt(np.sum(positions**2, axis=1)))
            print(f"  Final distance from origin: {final_distance:.3f} nm")
            print(f"  Maximum distance from origin: {max_distance:.3f} nm")

        # Create individual comparison plot for this potential
        print(f"\nCreating plots for {config['name']} potential...")
        fig = plot_integrator_comparison_nd(time, integrator_results, pot, config['name'], config['n_dims'])

        # Save individual plot
        filename = f"Dn_plots/integrator_comparison_nd_{config['name'].lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {filename}")

        # Plot potential surface
        pot_fig = plot_potential_surface_nd(pot, config['name'], config['n_dims'], config.get('x_range'))
        pot_filename = f"Dn_plots/potential_surface_nd_{config['name'].lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plt.savefig(pot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved potential plot: {pot_filename}")

        # Show plots
        plt.show()

    print(f"\n{'='*80}")
    print("ALL N-D TESTS COMPLETED")
    print(f"{'='*80}")


if __name__ == "__main__":
    # Define potential configurations to test
    potential_configs = [
        {
            'name': '2D Quadratic',
            'potential': Dn.Quadratic([[10.0, 0.0], [10.0, 0.0]]),  # k=10.0, center at origin for both dims
            'initial_x': np.array([1.0, 0.5]),
            'initial_v': np.array([0.0, 0.0]),
            'n_dims': 2,
            'x_range': np.linspace(-2, 2, 50),
            'debug_info': """2D QUADRATIC POTENTIAL
V(x,y) = 0.5 * k * [(x-x0)² + (y-y0)²]
Parameters: k=10.0, center=(0,0)
This is a 2D harmonic oscillator potential."""
        },
        {
            'name': '2D Quadratic (Anisotropic)',
            'potential': Dn.Quadratic([[10.0, 0.0], [5.0, 0.0]]),  # Different k for x,y
            'initial_x': np.array([1.0, 1.0]),
            'initial_v': np.array([0.5, -0.5]),
            'n_dims': 2,
            'x_range': np.linspace(-2, 2, 50),
            'debug_info': """2D ANISOTROPIC QUADRATIC POTENTIAL
V(x,y) = 0.5 * [kx*(x-x0)² + ky*(y-y0)²]
Parameters: kx=10.0, ky=5.0, center=(0,0)
Different spring constants in x and y directions."""
        },
        {
            'name': '3D Quadratic',
            'potential': Dn.Quadratic([[5.0, 0.0], [5.0, 0.0], [5.0, 0.0]]),  # k=5.0, center at origin for all dims
            'initial_x': np.array([1.0, 0.5, -0.5]),
            'initial_v': np.array([0.2, 0.1, 0.3]),
            'n_dims': 3,
            'x_range': np.linspace(-2, 2, 50),
            'debug_info': """3D QUADRATIC POTENTIAL
V(x,y,z) = 0.5 * k * [(x-x0)² + (y-y0)² + (z-z0)²]
Parameters: k=5.0, center=(0,0,0)
This is a 3D harmonic oscillator potential."""
        },
        {
            'name': '2D Circular Motion',
            'potential': Dn.Quadratic([[1.0, 0.0], [1.0, 0.0]]),  # Weak potential
            'initial_x': np.array([2.0, 0.0]),
            'initial_v': np.array([0.0, 2.0]),  # Perpendicular velocity for circular motion
            'n_dims': 2,
            'x_range': np.linspace(-3, 3, 50),
            'debug_info': """2D CIRCULAR MOTION TEST
V(x,y) = 0.5 * k * [x² + y²]
Parameters: k=1.0, center=(0,0)
Initial conditions set for circular/elliptical motion."""
        },
        {
            'name': '2D Quadratic (Off-center)',
            'potential': Dn.Quadratic([[8.0, 1.0], [8.0, -0.5]]),  # Off-center equilibrium
            'initial_x': np.array([0.0, 0.0]),
            'initial_v': np.array([0.1, 0.1]),
            'n_dims': 2,
            'x_range': np.linspace(-2, 3, 50),
            'debug_info': """2D OFF-CENTER QUADRATIC POTENTIAL
V(x,y) = 0.5 * [kx*(x-1.0)² + ky*(y+0.5)²]
Parameters: kx=8.0, ky=8.0, center=(1.0,-0.5)
Equilibrium position shifted from origin."""
        }
    ]
    
    # Run tests for all configurations
    run_integrator_tests_nd(potential_configs)
