import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import sys
    import ase
    import numpy as np
    import scipy as sp
    import matplotlib.pyplot as plt
    import marimo as mo

    from marimol import view_structure
    from ase.visualize import view


    from basics import get_cluster_2d, get_cluster_3d, show_cluster, get_potential, optimize_cluster, scan_distance

    sys.path.insert(0, '..')
    from kineticssandbox.system import system
    from kineticssandbox.potential import D1, Dn
    from kineticssandbox.integrator import Dn_integrator, Dn_stochastic

    return (
        D1,
        Dn_stochastic,
        get_cluster_2d,
        get_potential,
        np,
        optimize_cluster,
        plt,
        scan_distance,
        show_cluster,
        sp,
        system,
        view_structure,
    )


@app.cell
def _(get_cluster_2d, show_cluster, view_structure):
    # 2D Cluster setup
    sigma2rmin = 2.0**(1/6)
    cluster_radii = [0.0, 0.20*sigma2rmin]
    cluster_num_atoms = [1, 6]
    cluster_species = [
        ['C'],
        ['N', 'H', 'He', 'He', 'He', 'H']
    ]

    # Return an ASE atom cluster with atoms on circles at different radii
    positions_init, species_init = get_cluster_2d(
        cluster_radii,
        cluster_num_atoms,
        cluster_species,
    )

    # Show as 3d structure
    view_structure(
        show_cluster(
            positions_init,
            species_init,
        )
    )
    return positions_init, species_init


@app.cell
def _(D1, get_potential, np, positions_init, species_init):
    # Define atom masses per species in u
    # If mass is 0.0, the atom positions of these species are fixed
    cluster_masses = {
        'C': 0.0,
        'N': 50.0,
        'H': 18.0,
        'He': 0.0,
    }

    # Define atom pair properties in terms of:
    # [pair potential ('LJ1' or 'WCA1'), epsilon (kJ/mol), sigma (nm)]
    # Pair potentials:
    #   LJ1: 1-dimensional Lennard-Jones potential
    #   WCA1: 1-dimensional repulsive Weeks-Chandler-Andersen potential
    cluster_1d_potentials = {
        # 'C' to else
        ('C', 'N'):   [D1.LJ1, 10.0, 0.20],
        ('C', 'H'):   [D1.LJ1, 40.0, 0.20],
        #('C', 'He'):  [D1.LJ1, 20.0, 0.20],
        # 'N' to else
        ('N', 'H'):   [D1.LJ1,  1.0, 0.20],
        ('N', 'He'):  [D1.WCA1, 1.0, 0.20],
        # 'H' to else
        ('H', 'H'):   [D1.LJ1,  1.0, 0.20],
        ('H', 'He'):  [D1.WCA1, 1.0, 0.20],
        # 'He' to 'He' pair parameters arenot need as 'He' positions are fixed
        #('He', 'He'): [LJ1, 1.0, 0.20],
    }

    # Prepare atom pair potential but only for atom pairs with defined pair parameters.
    potential_md = get_potential(
        ndims=2,
        natoms=len(species_init),
        species=species_init,
        potential_parameters=cluster_1d_potentials,
    )

    # Test the potential function on the initial cluster
    # Regard converting from Angstrom to nanometer again
    energy_init = potential_md.potential(positions_init)
    forces_init = potential_md.force_ana(positions_init)
    print(
        f"The initial cluster energy is {energy_init:.3f} kJ/mol\n"
        + f"with a maximum force component of {np.amax(forces_init):.3f} kJ/mol/nm"
    )
    return cluster_masses, potential_md


@app.cell
def _(
    cluster_masses,
    np,
    optimize_cluster,
    positions_init,
    potential_md,
    show_cluster,
    species_init,
    view_structure,
):
    # Create an atom mask to select the atoms for the the positions should be minimized with respect to the energy
    # As mentioned above, all atoms with mass 0.0 are fixed
    mask_min = np.array(
        [
            cluster_masses[symbol] != 0.0
            for symbol in species_init
            for _ in range(2)
        ],
        dtype=bool
    )

    positions_min = optimize_cluster(
        positions=positions_init,
        potential=potential_md,
        mask=mask_min,
    )

    # Show as 3d structure
    view_structure(
        show_cluster(
            positions_min,
            species_init,
        )
    )
    return (positions_min,)


@app.cell
def _(np, plt, positions_min, potential_md, scan_distance):
    # Compute the 1D potential curve for the dissociation process using the optimized cluster
    distances = np.arange(0.15, 1.0, 0.001)
    energies = scan_distance(
        distances=distances,
        positions=positions_min,
        potential=potential_md,
        atom_i=0,
        atom_j=1,
        ndims=2,
    )

    # Store pair ponetial result
    np.savez(
        'diffusion_pair_potential_md_2d.npz',
        positions=positions_min,
        distances=distances,
        energies=energies,
    )

    # Shift energies tail to zero for pair potential
    pair_energies = energies - energies[-1]

    # Plot atom pair potential
    fig2, axs2 = plt.subplots()

    axs2.plot(
        distances,
        pair_energies,
        color='black',
        label=r'Atom Pair Potential'
    )

    axs2.set_ylim(1.1*np.min(pair_energies), -0.15*np.min(pair_energies))
    axs2.set_xlim(distances[0], 0.5)

    axs2.set_xlabel('Distance [nm]')
    axs2.set_ylabel('Potential [kJ/mol]')

    axs2.legend(loc='upper right', framealpha=1.0)

    plt.show()
    return


@app.cell
def _(
    Dn_stochastic,
    cluster_masses,
    np,
    positions_min,
    potential_md,
    show_cluster,
    species_init,
    system,
    view_structure,
):
    # Perform MD simulations of the cluster and store how long it takes for the 'N' atom to leave the cluster
    # Also define the 'N' atom distance threshold to the center 'C' atom, at which the system is counted as dissociated.
    md_diffusion_times = []
    md_threshold_distance = 1.0 # nm
    md_threshold_atom_i = 0 # 'C'
    md_threshold_atom_j = 1 # 'N'

    # First, the Langevin integrator parameters must be defined
    temperature = 300.0 # Kelvin, rememer <E_kin> = kB*T = 0.008314 kJ⋅mol-1⋅K-1 * T, <E_kin(300K)> = 2.5 kJ/mol
    friction = 1.0 # 1/ps
    time_step = 0.001 # ps

    # How many simulations will be running
    Nsimulation = 1_000

    # How many steps should each MD simulation run at maximum (each simulation is terminated when 'N' has left the cluster)
    Nsteps = 100_000 # *time_step = 100ps

    # Assign atom masses (for numerical reasons, set 0 masses to 1, their velocities will set to 0 each step)
    masses = np.array(
        [
            cluster_masses[symbol]
            for symbol in species_init
            for _ in range(2)
        ]
    )
    masses[masses == 0.0] = 1.0

    # Now, the full system (cluster + inegrator) can be defined
    system_2d = system.Dn(
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
            for _ in range(2)
        ],
        dtype=bool
    )

    # Import or define a function for initial velocity assignement
    # e.g. from ase.md.velocitydistribution import thermalize_momenta
    # or
    from basics import get_thermalized_velocities_nd
    _ = get_thermalized_velocities_nd(temperature, masses, scale_exactly=False, verbose=True)
    _ = get_thermalized_velocities_nd(temperature, masses, scale_exactly=True, verbose=True)

    # Sample one trajectory

    # Set cluster positions and velocities with respect to Boltzman distribution
    system_2d.x = positions_min
    system_2d.v = get_thermalized_velocities_nd(temperature, masses, scale_exactly=False, verbose=False)

    # Create trajectory position and velocity arrays and store the initial conditions
    trajectory_test_positions = np.zeros([Nsteps + 1, system_2d.x.shape[0]], dtype=float)
    trajectory_test_velocities = np.zeros([Nsteps + 1, system_2d.v.shape[0]], dtype=float)

    trajectory_test_positions[0, :] = system_2d.x.copy()
    trajectory_test_velocities[0, :] = system_2d.v.copy()

    # Run the simulation by iterating over time steps
    for _step in range(Nsteps):

        # Perform the time step by the Langevin integrator
        Dn_stochastic.BOAOB(
            system_2d,
            potential_md,
            fixed=mask_md,
        )

        # Store positions and velocities
        trajectory_test_positions[_step + 1, :] = system_2d.x.copy()
        trajectory_test_velocities[_step + 1, :] = system_2d.v.copy()

        # Student task, write a check function to test for dissociation:

        # Check for diffusion of atom 'N' and if the distance to the center 'C' atom is larger then the trheshold,
        # terminate the simulation.
        _distance = np.linalg.norm(
            (
                system_2d.x.reshape(-1, 2)[md_threshold_atom_i, :]
                - system_2d.x.reshape(-1, 2)[md_threshold_atom_j, :]
            )
        )

        if _distance > md_threshold_distance:
            print(f"Diffusion at {_step*time_step:6.3f} ps!")
            break

    # Show as 3d trajectory
    view_structure(
        [
            show_cluster(positions_i.reshape(-1, 2), species_init)
            for positions_i in trajectory_test_positions[0:_step + 1:100]
        ]
    )
    return (
        Nsimulation,
        Nsteps,
        get_thermalized_velocities_nd,
        mask_md,
        masses,
        md_diffusion_times,
        md_threshold_atom_i,
        md_threshold_atom_j,
        md_threshold_distance,
        system_2d,
        temperature,
        time_step,
    )


@app.cell
def _(
    Dn_stochastic,
    Nsimulation,
    Nsteps,
    get_thermalized_velocities_nd,
    mask_md,
    masses,
    md_diffusion_times,
    md_threshold_atom_i,
    md_threshold_atom_j,
    md_threshold_distance,
    np,
    positions_min,
    potential_md,
    system_2d,
    temperature,
    time_step,
):
    # Run simulation n time with different initial velocity distributions
    for irun in range(Nsimulation):

        # Reset cluster positions and velocities with respect to Boltzman distribution
        system_2d.x = positions_min
        system_2d.v = get_thermalized_velocities_nd(temperature, masses, scale_exactly=False, verbose=False)

        # Create trajectory position and velocity arrays and store the initial conditions
        trajectory_md_positions = np.zeros([Nsteps + 1, system_2d.x.shape[0]], dtype=float)
        trajectory_md_velocities = np.zeros([Nsteps + 1, system_2d.v.shape[0]], dtype=float)

        trajectory_md_positions[0, :] = system_2d.x.copy()
        trajectory_md_velocities[0, :] = system_2d.v.copy()

        # Run the simulation by iterating over time steps
        sucess = False
        for step in range(Nsteps):

            # Perform the time step by the Langevin integrator
            Dn_stochastic.BOAOB(
                system_2d,
                potential_md,
                fixed=mask_md,
            )

            # Store positions and velocities
            trajectory_md_positions[step + 1, :] = system_2d.x.copy()
            trajectory_md_velocities[step + 1, :] = system_2d.v.copy()

            # Check for diffusion of atom 'N' and if the distance to the center 'C' atom is larger then the trheshold,
            # terminate the simulation.
            distance = np.linalg.norm(
                (
                    system_2d.x.reshape(-1, 2)[md_threshold_atom_i, :]
                    - system_2d.x.reshape(-1, 2)[md_threshold_atom_j, :]
                )
            )

            if distance > md_threshold_distance:
                # Add diffusion time and print success
                md_diffusion_times.append(step*time_step)
                md_diffusion_mean = np.mean(md_diffusion_times)
                md_diffusion_rmsd = np.std(md_diffusion_times)
                print(
                    f"({irun + 1:3d}/{Nsimulation:d}) Diffusion at {step*time_step:6.3f} ps "
                    + f"({md_diffusion_mean:4.1f} +- {md_diffusion_rmsd:4.1f} ps)"
                )
                sucess = True
                break

        if not sucess:
            print(f"({irun + 1:d}/{Nsimulation:d}) No diffusion occured")

    # Show diffusion time
    md_diffusion_mean = np.mean(md_diffusion_times)
    md_diffusion_rmsd = np.std(md_diffusion_times)
    print(f"Mean diffusion time at {md_diffusion_mean:.3f} +- {md_diffusion_rmsd:.3f} ps")

    # Store results
    np.savetxt('diffusion_times_md_2d.dat', md_diffusion_times)
    return


@app.cell
def _(np, plt, sp):

    # Summarize the diffusion times as a survival probability by:
    md_diffusion_times_eval = np.loadtxt('diffusion_times_md_2d.dat')

    # Subtract the time until the first dissociation happens and declare this the equilibration time
    #md_diffusion_times_red = md_diffusion_times_eval - np.min(md_diffusion_times_eval)
    #md_diffusion_times_red = md_diffusion_times_red[md_diffusion_times_red > 0.0]
    md_diffusion_times_red = md_diffusion_times_eval

    # First, compute a time histogram of the diffusion times
    eval_tau_range = [0.0, 40.0] #ps
    eval_tau_nbins = 160 # 0.25 ps bins
    eval_tau_edges = np.linspace(*eval_tau_range, num=(eval_tau_nbins + 1))
    eval_tau_step = eval_tau_edges[1] - eval_tau_edges[0]
    eval_tau_centers = eval_tau_edges[:-1] + eval_tau_step/2.

    md_diffusion_times_histogram, _ = np.histogram(
        md_diffusion_times_red,
        bins=eval_tau_edges,
    )

    # Normalize the histogram to sum = 1
    md_diffusion_times_histogram = md_diffusion_times_histogram.astype(float)/np.sum(md_diffusion_times_histogram)

    # Second, convert to survival probability by summing up for each lifetime tau_i
    # the number of trajectories which were not dissociated yet
    md_diffusion_survival_propability = 1.0 - np.cumsum(md_diffusion_times_histogram)

    # Get time derivative
    der_md_diffusion_survival_propability = np.gradient(md_diffusion_survival_propability, eval_tau_step)

    # Fit linear function to the derivative
    mask_rate = np.logical_and(
        eval_tau_centers > 7.5,
        eval_tau_centers < 30.0,
    )

    md_diffusion_survival_rates = (
        -der_md_diffusion_survival_propability
        / md_diffusion_survival_propability
    )

    md_diffusion_survival_rate_mean = np.mean(md_diffusion_survival_rates[mask_rate])

    # Try exponential curve fit
    def exponential_decay(t, A, k):
        return A*np.exp(-k*t)
    def exponential_decay_delay(t, A, k, b):
        return A*np.exp(-(k*t)**b)

    popt, _ = sp.optimize.curve_fit(
        exponential_decay,
        eval_tau_centers[mask_rate],
        md_diffusion_survival_propability[mask_rate],
        p0=[1.0, 1.0]
    )

    popt_delay, _ = sp.optimize.curve_fit(
        exponential_decay_delay,
        eval_tau_centers,
        md_diffusion_survival_propability,
        p0=[1.0, 1.0, 1.0]
    )

    # Plot results
    fig1, axs1 = plt.subplots()

    axs1.scatter(
        eval_tau_centers,
        md_diffusion_survival_propability,
        color='grey',
        label=r'Survival Probability $S_A(t)$'
    )
    axs1.plot(
        eval_tau_centers,
        exponential_decay(eval_tau_centers, *popt),
        color='blue',
        label=r'Fit: $s_A(t) = $' + f'{popt[0]:.2f}$\cdot$exp(-{popt[1]:.2f}$t$)'
    )
    axs1.plot(
        eval_tau_centers,
        exponential_decay_delay(eval_tau_centers, *popt_delay),
        color='black',
        label=r'Fit: $f_A(t) = $' + f'{popt_delay[0]:.2f}$\cdot$exp(-({popt_delay[1]:.2f}$t)**${popt_delay[2]:.2f})'
    )
    taxs1 = axs1.twinx()
    taxs1.scatter(
        eval_tau_centers,
        der_md_diffusion_survival_propability,
        color='red',
        label=r'$d S_A(t) / d t$',
    )
    taxs1.scatter(
        eval_tau_centers,
        md_diffusion_survival_rates,
        color='green',
        label=r'$-(d S_A(t) / d t) / S_A(t)$'
    )

    taxs1.hlines(
        md_diffusion_survival_rate_mean,
        eval_tau_centers[0],
        eval_tau_centers[-1],
        linestyle='--',
        color='blue',
        label=r'$k_{AB}$ = ' + f'{md_diffusion_survival_rate_mean:.2f} 1/ps'
    )
    axs1.set_ylim(-0.1, 1.5)
    taxs1.set_ylim(-.2, 0.75)

    axs1.set_xlabel('Time t [ps]')
    axs1.set_ylabel('Survival Probability')
    taxs1.set_ylabel('Time Derivative [1/ps]', color='red')

    axs1.legend(loc='upper left', framealpha=1.0)
    taxs1.legend(loc='upper right', framealpha=1.0)

    print(r'Survival rate of $k_{AB}$ = ' + f'{md_diffusion_survival_rate_mean:.3f} 1/ps')

    plt.show()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
