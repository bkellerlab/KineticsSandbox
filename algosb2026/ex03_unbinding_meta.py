import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## AlgoSB 2026: Learning Protein Dynamics
    # Dynamic Reweighting for Unbinding Processes

    ### (3) Dynamic Reweigthing

    Now the original potential gets biased on the fly by the metadyanimcs algorithm to accelarate the unbinding process.
    """)
    return


@app.cell
def _():
    import sys

    import numpy as np
    import scipy as sp

    import matplotlib.pyplot as plt

    import marimo as mo
    from marimol import view_structure

    sys.path.insert(0, '../')
    from system import system
    from potential import D1, Dn
    from integrator import Dn_integrator, Dn_stochastic
    from utils.tools import (
        get_cluster_2d, get_cluster_3d,
        optimize_cluster, show_cluster,
        get_potential, get_biased_potential,
        scan_distance,
        get_thermalized_velocities_nd
    )

    return (
        D1,
        Dn_stochastic,
        get_biased_potential,
        get_thermalized_velocities_nd,
        mo,
        np,
        optimize_cluster,
        plt,
        scan_distance,
        show_cluster,
        system,
        view_structure,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As the bias is updated on the fly, we define the effective potential as a sum of the original potential and an empty bias potential.
    """)
    return


@app.cell
def _(D1, get_biased_potential, np):
    # Load the cluster
    cluster_data = np.load('lj_cluster_init.npz')
    positions_init = cluster_data['positions']
    species_init = cluster_data['species']
    natoms = cluster_data['natoms']
    ndims = cluster_data['ndims']


    # Define atom pair properties in terms of:
    # [pair potential ('LJ1' or 'WCA1'), epsilon (kJ/mol), sigma (nm)]
    # Pair potentials:
    #   LJ1: 1-dimensional Lennard-Jones potential
    #   WCA1: 1-dimensional repulsive Weeks-Chandler-Andersen potential
    cluster_1d_potentials = {
        # 'C' to else
        ('C', 'N'): [D1.LJ1, 12.0, 0.20],
        ('C', 'H'): [D1.LJ1, 40.0, 0.20],
        ('C', 'S'): [D1.LJ1, 80.0, 0.20],
        # 'N' to else
        ('N', 'H'): [D1.LJ1,  1.0, 0.20],
        ('N', 'S'): [D1.WCA1, 1.0, 0.20],
        # 'H' to else
        ('H', 'H'): [D1.LJ1,  1.0, 0.20],
        ('H', 'S'): [D1.WCA1, 1.0, 0.20],
        # 'S' to 'S' pair parameters are not needed as 'S' positions are always fixed later
        #('S', 'S'): [LJ1, 1.0, 0.20],
    }

    # Define no atom pair bias for now
    cluster_1d_bias = {}

    # Prepare atom pair potential but only for atom pairs with defined pair parameters.
    potential_biased = get_biased_potential(
        ndims=ndims,
        natoms=natoms,
        species=species_init,
        potential_parameters=cluster_1d_potentials,
        bias_parameters=cluster_1d_bias,
    )

    # Test the potential function on the initial cluster
    # Regard converting from Angstrom to nanometer again
    energies_init = potential_biased.potential(positions_init, verbose=True)
    forces_init = potential_biased.force_ana(positions_init)
    print(
        f"The initial cluster energy is {energies_init['effective']:.3f} kJ/mol "
        + f"(bias is {energies_init['bias']:.3f} kJ/mol)\n"
        + f"with a maximum force component of {np.amax(forces_init):.3f} kJ/mol/nm"
    )
    return natoms, ndims, positions_init, potential_biased, species_init


@app.cell
def _(
    ndims,
    np,
    optimize_cluster,
    positions_init,
    potential_biased,
    show_cluster,
    species_init,
    view_structure,
):
    # Define atom masses per species in u
    # If mass is 0.0, the atom positions of these species are fixed
    cluster_masses = {
        'C': 100.0,
        'N': 50.0,
        'H': 18.0,
        'S': 0.0,
    }

    # Create an atom mask to select the atoms for the the positions should be minimized with respect to the energy
    # As mentioned above, all atoms with mass 0.0 are fixed
    mask_min = np.array(
        [
            cluster_masses[symbol] != 0.0
            for symbol in species_init
            for _ in range(ndims)
        ],
        dtype=bool
    )

    positions_min = optimize_cluster(
        positions=positions_init,
        potential=potential_biased,
        mask=mask_min,
    )

    # Show as 3d structure
    view_structure(
        show_cluster(
            positions_min,
            species_init,
        )
    )
    return cluster_masses, positions_min


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <b> TASK </b> Before iterating over mutliple accelerated metadynamics simulations, test first the implementation of adding a Gaussian potential bias every n steps located at the respective C-N atom distance of that step with a certain Gaussian potential heigth and width. The second panel below shows the final bias potential over a potential scan of the C-N atom pair distance.

    Also store the accelerated sampled unbinding time and the rescaled unbinding time according to eq. (5.11).
    """)
    return


@app.cell
def _(
    D1,
    Dn_stochastic,
    cluster_masses,
    get_thermalized_velocities_nd,
    natoms,
    ndims,
    np,
    positions_min,
    potential_biased,
    show_cluster,
    species_init,
    system,
    view_structure,
):
    # First, the Langevin integrator parameters must be defined
    temperature = 300.0 # Kelvin, rememer <E_kin> = kB*T = 0.008314 kJ⋅mol-1⋅K-1 * T, <E_kin(300K)> = 2.5 kJ/mol
    friction = 1.0 # 1/ps
    time_step = 0.010 # ps

    kB_kJmolK = 0.008314462618
    beta = 1./(kB_kJmolK*temperature)

    # Assign atom masses (for numerical reasons, set 0 masses to 1, their velocities will set to 0 each step)
    masses = np.array(
        [
            cluster_masses[symbol]
            for symbol in species_init
            for _ in range(ndims)
        ]
    )
    masses[masses == 0.0] = 1.0

    # Now, the full system (cluster + inegrator) can be defined
    cluster = system.Dn(
        m=masses,
        x=positions_min,
        v=np.zeros_like(positions_min),
        T=temperature,
        xi=friction,
        dt=time_step,
        h=None,
    )

    # Use the same atom mask as in the minimization step to fix certain cluster atom positions
    mask_md = np.array(
        [
            cluster_masses[symbol] == 0.0
            for symbol in species_init
            for _ in range(ndims)
        ],
        dtype=bool
    )

    # Perform MD simulations of the cluster and store how long it takes for the 'N' atom to leave the cluster
    # Also define the 'N' atom distance threshold to the center 'C' atom, at which the system is counted as dissociated.
    _md_unbinding_times = []
    md_threshold_distance = 1.0 # nm
    md_threshold_atom_i = 0 # 'C'
    md_threshold_atom_j = 1 # 'N'

    # Define here the metadynamics parameters such as the collective variable CV, add-frequency, the add-potential
    # and its parameters
    meta_cv_atom_i = 0 # 'C'
    meta_cv_atom_j = 0 # 'N'
    meta_add_steps = 0 # every ?00 steps or 0.? ps
    meta_potential = D1.GeneralGaussian # General Gaussian function alpha * exp(c*(x - a)**2)
    meta_parameter = [None, 100., 0.00] # a, c = 1/(2*sigma**2) with sigma in [nm], alpha in [kJ/mol]
    # (parameter a for the location will be updated during the simulation)

    # How many steps should the MD simulation run
    Nsteps_bias = 10_000 # Nsteps*time_step = 100ps

    # Set cluster positions and velocities with respect to Boltzman distribution
    cluster.x = positions_min
    cluster.v = get_thermalized_velocities_nd(temperature, masses, scale_exactly=True)

    # Create trajectory position and velocity arrays and store the initial conditions
    trajectory_test_positions = np.zeros([Nsteps_bias + 1, natoms, ndims], dtype=float)
    trajectory_test_velocities = np.zeros([Nsteps_bias + 1, natoms, ndims], dtype=float)
    trajectory_test_bias = np.zeros(Nsteps_bias + 1, dtype=float)

    trajectory_test_positions[0, :] = cluster.x.reshape(natoms, ndims)
    trajectory_test_velocities[0, :] = cluster.v.reshape(natoms, ndims)
    trajectory_test_bias[0] = potential_biased.potential(
        cluster.x,
        verbose=True,
    )['bias'] # dictionary with items for 'original', 'bias', 'effective' (= 'original' + 'bias')

    # Remove previous potential bias, if it already exists from a previous run
    potential_biased.remove_bias()

    # Run the simulation by iterating over time steps
    for _step in range(Nsteps_bias):

        # Perform the time step by the Langevin integrator
        Dn_stochastic.BOAOB(
            cluster,
            potential_biased,
            fixed=mask_md,
        )

        # Store positions and velocities
        trajectory_test_positions[_step + 1] = cluster.x.reshape(natoms, ndims)
        trajectory_test_velocities[_step + 1] = cluster.v.reshape(natoms, ndims)

        # Store bias potential
        trajectory_test_bias[_step + 1] = potential_biased.potential(
            cluster.x,
            verbose=True,
        )['bias']

        # Check for unbinding of atom 'N' and if the distance to the center 'C' atom is larger then the trheshold,
        # terminate the simulation.
        _distance = np.linalg.norm(
            (
                cluster.x.reshape(natoms, ndims)[md_threshold_atom_i]
                - cluster.x.reshape(natoms, ndims)[md_threshold_atom_j]
            )
        )

        if _distance > md_threshold_distance:
            _md_unbinding_times.append(_step*time_step)

            break

    # Show as 3d trajectory
    view_structure(
        [
            show_cluster(positions_i.reshape(-1, ndims), species_init)
            for positions_i in np.concatenate(
                (
                    trajectory_test_positions[0:_step + 1:10],
                    trajectory_test_positions[_step + 1][np.newaxis, :],
                )
            )
        ],
        trajectory_slider=True,
        measuring_tool=True,
        recording_tools=True,
    )
    return


@app.cell
def _(ndims, np, plt, positions_min, potential_biased, scan_distance):
    # Plot the accumalated bias potential
    distances = np.arange(0.15, 1.0, 0.001)
    energies_biased = scan_distance(
        distances=distances,
        positions=positions_min,
        potential=potential_biased,
        atom_i=0,
        atom_j=1,
        ndims=ndims,
        verbose=True
    )

    # Get energy contributions
    energies_original = energies_biased['original']
    energies_bias = energies_biased['bias']
    energies_effective = energies_biased['effective']

    # Shift original energies tail to zero and shift the effective by the same value
    pair_energies_original = energies_original - energies_original[-1]
    pair_energies_effective = energies_effective - energies_effective[-1]

    # Plot atom pair potential
    fig1, axs1 = plt.subplots()

    axs1.plot(
        distances,
        pair_energies_original,
        color='black',
        label=r'Original Atom Pair Potential'
    )
    axs1.plot(
        distances,
        energies_bias,
        linestyle='--',
        color='orange',
        label=r'Bias Atom Pair Potential'
    )
    axs1.plot(
        distances,
        pair_energies_effective,
        color='red',
        label=r'Effective Atom Pair Potential'
    )

    min_pair_energy = np.min([np.min(pair_energies_effective), np.min(pair_energies_original)])
    axs1.set_ylim(1.1*min_pair_energy, 1.5*np.max(energies_bias))
    axs1.set_xlim(distances[0], 1.0)

    axs1.set_xlabel('Distance [nm]')
    axs1.set_ylabel('Potential [kJ/mol]')

    axs1.legend(loc='lower right', framealpha=1.0)

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <b> TASK </b> Iterate over multiple accelerated metadynamics simulations and store the accelerated and reweigthed unbinding time for the subsequent evaluation as done before.
    """)
    return


@app.cell
def _(md_unbinding_times, md_unbinding_times_scaled, np):
    ...

    # Store results
    np.savetxt(
        'unbinding_times_meta.dat',
        np.vstack((md_unbinding_times, md_unbinding_times_scaled)).T
    )
    return


@app.cell
def _(np):
    # Load unbinding times
    md_unbinding_times_eval, md_unbinding_times_scaled_eval = np.loadtxt('unbinding_times_meta.dat').T

    ...
    return


if __name__ == "__main__":
    app.run()
