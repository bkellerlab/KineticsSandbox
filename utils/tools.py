import os
import sys
import ase
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import marimo as mo

from marimol import view_structure
from ase.visualize import view

sys.path.insert(0, '..')

from kineticssandbox.system import system
from kineticssandbox.potential import D1, Dn
from kineticssandbox.integrator import Dn_integrator, Dn_stochastic


def fibonacci_lattice_on_sphere(
    num_atoms: int,
    radius: float,
) -> list[float]:
    """
    Get a Cartesian representation of a Fibonacci lattice on a spherical
    surface. See, e.g.: https://observablehq.com/@meetamit/fibonacci-lattices
    """

    # Special case for just on particle, set at (radius, 0, 0)
    if num_atoms == 1:
        return [[radius, 0.0, 0.0]]

    # Golden ratio
    phi = np.pi * (3.0 - np.sqrt(5.0))  

    # Get (x,y,z)-coordinates for the distributed particles on a sphere
    positions = []
    for ii in range(num_atoms):
    
        # Get unit radius from a projection along y
        y = 1.0 - 2.0*(ii/float(num_atoms - 1))
        ry = np.sqrt(np.max([0.0, 1.0 - y**2]))

        # Get golden ratio increment
        theta = ii*phi

        # Determine remaining Cartesian coordinates
        x = ry*np.cos(theta)
        z = ry*np.sin(theta)

        # Scale to actual radius
        positions.append([radius*x, radius*y, radius*z])

    return positions


def get_cluster_2d(
    cluster_radii: list[float],
    cluster_num_atoms: list[int],
    cluster_species: list[str],
) -> [np.ndarray[float], list[str]]:
    """
    Create a cluster by equally distribute particles on circles of different
    circumferences.
    """
    
    
    # Get (x,y)-coordinates for the distributed particles
    atom_positions = []
    atom_species = []
    start_angle = 0.0
    for radius, num_atoms, species in zip(
        cluster_radii, cluster_num_atoms, cluster_species
    ):
        
        # Set particles on the circle
        for ii in range(num_atoms):
            angle = start_angle + ii*2*np.pi/num_atoms
            atom_positions.append(
                [
                    radius*np.cos(angle),
                    radius*np.sin(angle),
                ]
            )

        # Add species
        atom_species += list(species)
        
        # Modify start angle for gap-staggered particles
        start_angle = np.pi/num_atoms

    return np.array(atom_positions), atom_species


def get_cluster_3d(
    cluster_radii: list[float],
    cluster_num_atoms: list[int],
    cluster_species: list[str],
) -> ase.Atoms:
    """
    Create a cluster by equally distribute particles on circles of different 
    circumferences.
    """

    # Get (x,y,z)-coordinates for the distributed particles
    atom_positions = []
    atom_species = []
    for radius, num_atoms, species in zip(
        cluster_radii, cluster_num_atoms, cluster_species
    ):

        # Get particles on a sphere
        atom_positions += list(
            fibonacci_lattice_on_sphere(
                num_atoms,
                radius
            )
        )

        # Add species
        atom_species += list(species)

    # Create as an ASE Atoms object (convert coordinates from GROMACS nm to 
    # ASE Angstrom)
    cluster_ase = ase.Atoms(
        atom_species,
        positions=10.*np.array(atom_positions)
    )

    return cluster_ase


def show_cluster(
    positions: np.ndarray[float],
    species: list[str]
) -> dict[str, np.ndarry[float] | list[str]]:

    # Create as an ASE Atoms object (convert coordinates from GROMACS nm to 
    # ASE Angstrom)
    if positions.shape[1] == 2:
        positions_3d = np.zeros([positions.shape[0], 3], dtype=float)
        positions_3d[:, :2] = positions
    elif positions.shape[1] == 3:
        positions_3d = positions
    else:
        raise ValueError(
            "Positons input must have dimension of (Natoms, 2) or (Natoms, 3)!"
        )

    return {
        'positions': (10.*positions_3d).tolist(),
        'species': list(species),
    }


def get_potential(
    ndims: int,
    natoms: int,
    species: list[str],
    potential_parameters: dict[tuple, tuple[D1.D1, '*parameters']],
) -> Dn.Dn:
    """
    Prepare atom pair potential and indices list but only for atom pairs with
    defined pair parameters.
    """
    
    # Initialize pair potential and indices lists
    pair_potentials = []
    pair_indices = []
    
    # Here we iterate over all combinations of an upper triangular matrix part.
    for ii in range(natoms - 1):
        for jj in range(ii + 1, natoms):

            # Check for potential parameters,
            # first by tuple pairs of atom indices
            # and second by tuple pairs of atom species
            pair_parameters = None
            if (ii, jj) in potential_parameters:
                pair_parameters = potential_parameters[(ii, jj)]
            elif (jj, ii) in potential_parameters:
                pair_parameters = potential_parameters[(jj, ii)]
            elif (species[ii], species[jj]) in potential_parameters:
                pair_parameters = potential_parameters[
                    (species[ii], species[jj])
                ]
            elif (species[jj], species[ii]) in potential_parameters:
                pair_parameters = potential_parameters[
                    (species[jj], species[ii])
                ]

            # Add pair potential if found
            if pair_parameters is not None:
                pair_potentials.append(
                    pair_parameters[0](pair_parameters[1:])
                )
                pair_indices.append([ii, jj])

    # Convert and transpose pair indices list to an array for convenient indexing
    pair_indices = np.array(pair_indices).T.reshape(
        2,
        len(pair_potentials),
    )

    # Create the potential instance to compute cluster energies or atom forces
    potential_3d = Dn.PairPotentialSum(
        ndim=ndims,
        natoms=natoms,
        pair_potentials=pair_potentials,
        pair_indices=pair_indices,
    )

    return potential_3d


def get_bias_potential(
    ndims: int,
    natoms: int,
    species: list[str],
    potential_parameters: dict[tuple, tuple[D1.D1, '*parameters']],
    bias_parameters: dict[tuple, tuple[D1.D1, '*parameters']],
) -> Dn.Dn:
    """
    Prepare atom pair potential and indices list but only for atom pairs with
    defined pair parameters.
    """
    
    # Create the potential instance to compute cluster energies or atom forces
    potential_3d = get_potential(
        ndims,
        natoms,
        species,
        potential_parameters,
    )

    # Create the bias potential instance
    bias_3d = get_potential(
        ndims,
        natoms,
        species,
        bias_parameters,
    )

    # Combine to one potential class
    potential_and_bias = Dn.BiasedPairPotentialSum(potential_3d, bias_3d)

    return potential_and_bias


def optimize_cluster(
    positions: np.ndarray[float],
    potential: Dn.Dn,
    mask: np.ndarray[int],
) -> np.ndarray[float]:
    """
    As the maximum absolute force component is very high, we need to minimize
    the atom positions 'carefully' with respect to the cluster energy as we do
    not want the cluster to disiciate right here. For this we can use the basic
    (but powerful) scipy 'minimize' function.
    """

    from scipy.optimize import minimize

    # Create an auxiliary function for the energy and force evaluation
    def eval_potential(
        positions_to_minimize: np.ndarray[float],
        positions_all: np.ndarray[float] = positions,
        mask: np.ndarray[bool] = mask
    ) -> float:

        # Combine initial positions with current position guess
        positions_step = positions_all.copy().flatten()
        positions_step[mask] = positions_to_minimize

        # Compute potential energy
        return potential.potential(positions_step)

    def eval_gradient(
        positions_to_minimize: np.ndarray[float],
        positions_all: np.ndarray[float] = positions,
        mask: np.ndarray[bool] = mask
    ) -> np.ndarray[float]:

        # Combine initial positions with current position guess
        positions_step = positions_all.copy().flatten()
        positions_step[mask] = positions_to_minimize

        # Compute the gradient (= negative forces)
        return -potential.force_ana(positions_step)[mask]


    # Perform structure minimization
    result = minimize(
        eval_potential,
        positions.copy().flatten()[mask],
        method='BFGS',
        jac=eval_gradient,
        tol=0.01
    )

    # Get optimized cluster positions and compute minimized energy and forces
    shape_positions = positions.shape
    positions_min = positions.copy().flatten()
    positions_min[mask] = result.x.copy()
    energy_min = potential.potential(positions_min)
    forces_min = potential.force_ana(positions_min)
    forces_min[np.invert(mask)] = 0.0
    print(
        f"The optimized cluster energy is {energy_min:.3f} kJ/mol\n"
        + f"with a maximum force component of {np.amax(forces_min):.3f} kJ/mol/nm"
    )

    return positions_min.reshape(shape_positions)


def scan_distance(
    distances: list[float],
    positions: np.ndarray[float],
    potential: Dn.Dn,
    atom_i: int = 0,
    atom_j: int = 1,
    ndims: int = 3,
    verbose: bool = False
) -> np.ndarray[float]:
    """
    Compute the 1D potential curve for different distances between atom i and j.
    """
    
    # Get the atom pair vector
    vector = (
        positions.reshape(-1, ndims)[atom_j]
        - positions.reshape(-1, ndims)[atom_i]
    )
    vector /= np.linalg.norm(vector)

    # Scan over distances
    if verbose:
        energies_scan = np.zeros([distances.shape[0], 2], dtype=float)
    else:
        energies_scan = np.zeros_like(distances)
    for ii, distance in enumerate(distances):

        # Set atom pair distance
        positions_scan = positions.copy().reshape(-1, ndims)
        positions_scan[atom_j] = positions_scan[atom_i] + distance*vector

        # Compute energy
        energies_scan[ii] = potential.potential(
            positions_scan.flatten(),
            verbose=verbose
        )

    return np.array(energies_scan, dtype=float)


def get_thermalized_velocities_nd(
    temperature_K: float,
    masses: list[float],
    scale_exactly: bool = True,
    verbose: bool = True,
) -> np.ndarray[float]:

    # Mass list with masses in kg for each degree of freedom
    u2kg = 1.66053906892E-27
    masses_kg = np.array(masses).flatten()*u2kg

    # Draw random numbers with respect to normal distribution (Gauss-like)
    xi = np.random.standard_normal(len(masses_kg))

    # Predict momenta in kg*m/s
    kB_JK = 1.380649E-23
    momenta = xi*np.sqrt(masses_kg*kB_JK*temperature_K)

    # Convert to velocities in m/s
    velocities = momenta/masses_kg

    # Check temperature
    kinetic_energy = 0.5*np.sum(masses_kg*velocities**2)
    predicted_temperature = (2.0*kinetic_energy)/(len(masses_kg)*kB_JK)
    if scale_exactly:
        velocities *= np.sqrt(temperature_K / predicted_temperature)
        scaled_kinetic_energy = 0.5*np.sum(masses_kg*velocities**2)
        scaled_temperature = (2.0*scaled_kinetic_energy)/(len(masses_kg)*kB_JK)
        if verbose:
            print(
                "Randomly assigned velocities of temperature "
                + f"{predicted_temperature:.1f} K are scaled to "
                + f"{scaled_temperature:.1f} K!"
            )
    else:
        if verbose:
            print(
               f"Randomly assigned velocities yield a temperature of "
               + f"{predicted_temperature:.1f} K!"
            )

    # Return velocities in nm/ps
    return 1.E9/1.E12*velocities
