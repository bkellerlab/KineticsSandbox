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
        ('C', 'N'): [D1.LJ1, 10.0, 0.20],
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
    time_step = 0.001 # ps

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
        h=0.001,
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

    # Sample one trajectory

    # Student task:

    # Perform MD simulations of the cluster and store how long it takes for the 'N' atom to leave the cluster
    # Also define the 'N' atom distance threshold to the center 'C' atom, at which the system is counted as dissociated.
    _md_unbinding_times = []
    md_threshold_distance = 1.0 # nm
    md_threshold_atom_i = 0 # 'C'
    md_threshold_atom_j = 1 # 'N'

    # Define here the metadynamics parameters such as the collective variable CV, add-frequency, the add-potential
    # and its parameters
    """
    # Student task:

    _md_unbinding_times_scaled = []

    meta_cv_atom_i = 0 # 'C'
    meta_cv_atom_j = 1 # 'N'
    meta_add_steps = 100 # every 100 steps or 0.1 ps
    meta_potential = D1.GeneralGaussian # General Gaussian function alpha * exp(c*(x - a)**2)
    meta_parameter = [None, 100., 0.1] # a, c = sqrt(sigma/2) with sigma in [nm], alpha in [kJ/mol]
    # (parameter a for the location will be updated during the simulation)

    """

    # How many steps should the MD simulation run
    Nsteps_bias = 100_000 # Nsteps*time_step = 100ps

    # Set cluster positions and velocities with respect to Boltzman distribution
    cluster.x = positions_min
    cluster.v = get_thermalized_velocities_nd(temperature, masses, scale_exactly=True)

    # Create trajectory position and velocity arrays and store the initial conditions
    trajectory_test_positions = np.zeros([Nsteps_bias + 1, cluster.x.shape[0]], dtype=float)
    trajectory_test_velocities = np.zeros([Nsteps_bias + 1, cluster.v.shape[0]], dtype=float)
    trajectory_test_bias = np.zeros(Nsteps_bias + 1, dtype=float)

    trajectory_test_positions[0, :] = cluster.x.copy()
    trajectory_test_velocities[0, :] = cluster.v.copy()
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
        trajectory_test_positions[_step + 1, :] = cluster.x.copy()
        trajectory_test_velocities[_step + 1, :] = cluster.v.copy()

        # Store bias potential
        trajectory_test_bias[_step + 1] = potential_biased.potential(
            cluster.x,
            verbose=True,
        )['bias']

        """
        # Student task:

        # Add metadynamic bias if when time is come
        if not (_step + 1) % meta_add_steps:

            _distance = np.linalg.norm(
                (
                    cluster.x.reshape(natoms, ndims)[meta_cv_atom_i, :]
                    - cluster.x.reshape(natoms, ndims)[meta_cv_atom_j, :]
                )
            )

            potential_biased.add_bias(
                bias_potentials=[
                    meta_potential([_distance, *meta_parameter[1:]])
                ],
                bias_indices=np.array([[meta_cv_atom_i], [meta_cv_atom_j]])
            )
        """

        # Check for unbinding of atom 'N' and if the distance to the center 'C' atom is larger then the trheshold,
        # terminate the simulation.
        _distance = np.linalg.norm(
            (
                cluster.x.reshape(natoms, ndims)[md_threshold_atom_i, :]
                - cluster.x.reshape(natoms, ndims)[md_threshold_atom_j, :]
            )
        )

        if _distance > md_threshold_distance:
            _md_unbinding_times.append(_step*time_step)
            """
            # Student task:
            _md_unbinding_times_scaled.append(
                time_step*np.sum(np.exp(+beta*trajectory_test_bias[:_step + 1]))
            )
            print(
                f"Accelerated unbinding at {_md_unbinding_times[-1]:6.3f} ps, "
                + f"which corresponds to an unbinding time of {_md_unbinding_times_scaled[-1]:6.3f} ps!"
            )
            """
            break

    # Show as 3d trajectory
    view_structure(
        [
            show_cluster(positions_i.reshape(-1, ndims), species_init)
            for positions_i in np.concatenate(
                (
                    trajectory_test_positions[0:_step + 1:100],
                    trajectory_test_positions[_step + 1].reshape(1, -1),
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
    """
    # Student task:

    md_unbinding_times = []
    md_unbinding_times_scaled = []

    # How many simulations will be performed
    Nsimulation = 1_000

    # How many steps should the MD simulation run
    Nsteps = 100_000 # Nsteps*time_step = 100ps

    # Run simulation n time with different initial velocity distributions
    for irun in range(Nsimulation):

        # Remove previous potential bias, if it already exists from a previous run
        potential_biased.remove_bias()

        # Reset cluster positions and velocities with respect to Boltzman distribution
        cluster.x = positions_min
        cluster.v = get_thermalized_velocities_nd(temperature, masses, scale_exactly=True)

        # Create trajectory position and velocity arrays and store the initial conditions
        trajectory_md_positions = np.zeros([Nsteps + 1, cluster.x.shape[0]], dtype=float)
        trajectory_md_velocities = np.zeros([Nsteps + 1, cluster.v.shape[0]], dtype=float)
        trajectory_md_bias = np.zeros(Nsteps + 1, dtype=float)

        trajectory_md_positions[0, :] = cluster.x.copy()
        trajectory_md_velocities[0, :] = cluster.v.copy()

        # Run the simulation by iterating over time steps
        sucess = False
        for _step in range(Nsteps):

            # Perform the time step by the Langevin integrator
            Dn_stochastic.BOAOB(
                cluster,
                potential_biased,
                fixed=mask_md,
            )

            # Store positions and velocities
            trajectory_md_positions[_step + 1, :] = cluster.x.copy()
            trajectory_md_velocities[_step + 1, :] = cluster.v.copy()

            # Store bias potential
            trajectory_md_bias[_step + 1] = potential_biased.potential(
                cluster.x,
                verbose=True,
            )['bias']
    
            # Add metadynamic bias if when time is come
            if not (_step + 1) % meta_add_steps:
    
                _distance = np.linalg.norm(
                    (
                        cluster.x.reshape(natoms, ndims)[meta_cv_atom_i, :]
                        - cluster.x.reshape(natoms, ndims)[meta_cv_atom_j, :]
                    )
                )
    
                potential_biased.add_bias(
                    bias_potentials=[
                        meta_potential([_distance, *meta_parameter[1:]])
                    ],
                    bias_indices=np.array([[meta_cv_atom_i], [meta_cv_atom_j]])
                )

            # Check for unbinding of atom 'N' and if the distance to the center 'C' atom is larger then the trheshold,
            # terminate the simulation.
            distance = np.linalg.norm(
                (
                    cluster.x.reshape(natoms, ndims)[md_threshold_atom_i, :]
                    - cluster.x.reshape(natoms, ndims)[md_threshold_atom_j, :]
                )
            )

            if distance > md_threshold_distance:
                # Add unbinding time and print success
                md_unbinding_times.append(_step*time_step)
                md_unbinding_mean = np.mean(md_unbinding_times)
                md_unbinding_rmsd = np.std(md_unbinding_times)
                md_unbinding_times_scaled.append(
                    time_step*np.sum(np.exp(+beta*trajectory_test_bias[:_step + 1]))
                )
                md_unbinding_mean_scaled = np.mean(md_unbinding_times_scaled)
                md_unbinding_rmsd_scaled = np.std(md_unbinding_times_scaled)
                print(
                    f"({irun + 1:3d}/{Nsimulation:d}) Accelerated unbinding at {_step*time_step:6.3f} ps "
                    + f"({md_unbinding_mean:4.1f} +- {md_unbinding_rmsd:4.1f} ps), "
                    + f"which corresponds to an unbinding time of {md_unbinding_times_scaled[-1]:6.3f} ps "
                    + f"({md_unbinding_mean_scaled:4.1f} +- {md_unbinding_rmsd_scaled:4.1f} ps)."
                )
                sucess = True
                break

        if not sucess:
            print(f"({irun + 1:d}/{Nsimulation:d}) No unbinding occured")

    # Show unbinding time
    md_unbinding_mean = np.mean(md_unbinding_times)
    md_unbinding_rmsd = np.std(md_unbinding_times)
    print(f"Mean accelerated unbinding time at {md_unbinding_mean:.3f} +- {md_unbinding_rmsd:.3f} ps")
    print(f"Mean reweighted unbinding time at {md_unbinding_mean:.3f} +- {md_unbinding_rmsd:.3f} ps")
    """

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

    """
    # Student task:

    # Student task:

    # First, compute a time histogram of the unbinding times
    eval_tau_range = [0.0, 40.0] #ps
    eval_tau_nbins = 160 # 0.25 ps bins
    eval_tau_edges = np.linspace(*eval_tau_range, num=(eval_tau_nbins + 1))
    eval_tau_step = eval_tau_edges[1] - eval_tau_edges[0]
    eval_tau_centers = eval_tau_edges[:-1] + eval_tau_step/2.

    md_unbinding_times_histogram, _ = np.histogram(
        md_unbinding_times_scaled_eval,
        bins=eval_tau_edges,
    )

    # Normalize the histogram to sum = 1
    md_unbinding_times_histogram = md_unbinding_times_histogram.astype(float)/np.sum(md_unbinding_times_histogram)

    # Second, convert to survival probability by summing up for each lifetime tau_i
    # the number of trajectories which were not dissociated yet
    md_unbinding_survival_propability = 1.0 - np.cumsum(md_unbinding_times_histogram)

    # Get time derivative
    der_md_unbinding_survival_propability = np.gradient(md_unbinding_survival_propability, eval_tau_step)

    # Fit linear function to the derivative
    mask_rate = np.logical_and(
        eval_tau_centers > 2.5,
        eval_tau_centers < 7.5,
    )

    md_unbinding_survival_rates = (
        -der_md_unbinding_survival_propability
        / md_unbinding_survival_propability
    )

    md_unbinding_survival_rate_mean = np.mean(md_unbinding_survival_rates[mask_rate])

    # Try exponential curve fit
    def exponential_decay(t, A, k):
        return A*np.exp(-k*t)
    def exponential_decay_delay(t, A, k, b):
        return A*np.exp(-(k*t)**b)

    popt, _ = sp.optimize.curve_fit(
        exponential_decay,
        eval_tau_centers[mask_rate],
        md_unbinding_survival_propability[mask_rate],
        p0=[1.0, 1.0]
    )

    popt_delay, _ = sp.optimize.curve_fit(
        exponential_decay_delay,
        eval_tau_centers,
        md_unbinding_survival_propability,
        p0=[1.0, 1.0, 1.0]
    )

    # Plot results
    fig2, axs2 = plt.subplots()

    axs2.scatter(
        eval_tau_centers,
        md_unbinding_survival_propability,
        color='grey',
        label=r'Survival Probability $S_A(t)$'
    )
    axs2.plot(
        eval_tau_centers,
        exponential_decay(eval_tau_centers, *popt),
        color='blue',
        label=r'Fit: $s_A(t) = $' + f'{popt[0]:.2f}$\cdot$exp(-{popt[1]:.2f}$t$)'
    )
    axs2.plot(
        eval_tau_centers,
        exponential_decay_delay(eval_tau_centers, *popt_delay),
        color='black',
        label=r'Fit: $f_A(t) = $' + f'{popt_delay[0]:.2f}$\cdot$exp(-({popt_delay[1]:.2f}$t)**${popt_delay[2]:.2f})'
    )
    taxs2 = axs2.twinx()
    taxs2.scatter(
        eval_tau_centers,
        der_md_unbinding_survival_propability,
        color='red',
        label=r'$d S_A(t) / d t$',
    )
    taxs2.scatter(
        eval_tau_centers,
        md_unbinding_survival_rates,
        color='green',
        label=r'$-(d S_A(t) / d t) / S_A(t)$'
    )

    taxs2.hlines(
        md_unbinding_survival_rate_mean,
        eval_tau_centers[0],
        eval_tau_centers[-1],
        linestyle='--',
        color='blue',
        label=r'$k_{AB}$ = ' + f'{md_unbinding_survival_rate_mean:.2f} 1/ps'
    )
    axs2.set_ylim(-0.1, 1.5)
    taxs2.set_ylim(-.3, 0.75)

    axs2.set_xlabel('Time t [ps]')
    axs2.set_ylabel('Survival Probability')
    taxs2.set_ylabel('Time Derivative [1/ps]', color='red')

    axs2.legend(loc='upper left', framealpha=1.0)
    taxs2.legend(loc='upper right', framealpha=1.0)

    print(r'Survival rate by exponential fit: $k_{AB,1}$ = ' + f'{popt[1]:.3f} 1/ps')
    print(r'Survival rate by linear fit: $k_{AB,2}$ = ' + f'{md_unbinding_survival_rate_mean:.3f} 1/ps')

    plt.show()
    """
    return


if __name__ == "__main__":
    app.run()
