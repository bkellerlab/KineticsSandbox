import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## AlgoSB 2026: Learning Protein Dynamics
    # Dynamic Reweighting for Unbinding Processes

    <figure>
      <img src="public/cluster_intro.png" width="600" />
      <figcaption>Figure 1: Sketch of a 2D cluster and the unbinding process.</figcaption>
    </figure>

    In this exercise, we simulate the unbinding process of a target atom from the model of a Lennard-Jones (LJ) cluster.
    First, we introduce the this version of setting up a cluster in either 2D or 3d Cartesian coordinate system.
    Then we define the LJ pair potential parameters and set up the potential instance to compute the potential energy and forces acting on the cluster atom for any given position array.

    With both, the cluster and potential (force field) instance, we show how to perform a structure optimization and to simulate the unbinding process by classical molecular dynamics using a stochasting integrator with constant temperature.

    For the start, let us import all the required modules for this exercise. Note the <code> sys.path.insert(0, '../') </code> command which adds the parent directory (<code> '../' </code>) to the beginning (<code> 0 </code>) of the <code> PYTHONPATH </code> environmental variable, which is a list of directories for Python to search for modules.
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
        get_potential,
        scan_distance,
        get_thermalized_velocities_nd
    )


    return (
        D1,
        Dn_stochastic,
        get_cluster_2d,
        get_cluster_3d,
        get_potential,
        get_thermalized_velocities_nd,
        mo,
        np,
        optimize_cluster,
        plt,
        scan_distance,
        show_cluster,
        sp,
        system,
        view_structure,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here, 2D clusters does not just mean an initial configuration of atoms within the same plane, but rather that all atom movement only happens on this (x,y)-plane. Even though it is a significant simplificication, the physics stays the same and it increased the visual clarity.

    In the 2D cluster, a number of atoms <code> cluster_num_atoms </code> are equidistantly distributed on circles with radii <code> cluster_radii </code>. Each atom is assigned a species label which has no relation to the real chemical symbol as all atom and atom pair properties such as atom mass and interaction strength are later individually defined per species (only the colors when displayed are determined by the species).
    """)
    return


@app.cell
def _(get_cluster_2d, show_cluster, view_structure):
    sigma2rmin = 2.0**(1/6)

    # 2D Cluster setup
    cluster_radii_2d = [0.0, 0.20*sigma2rmin]
    cluster_num_atoms_2d = [1, 6]
    cluster_species_2d = [
        ['C'],
        ['N', 'H', 'S', 'S', 'S', 'H'],
    ]

    # Return an ASE atom cluster with atoms on circles at different radii
    positions_init_2d, species_init_2d = get_cluster_2d(
        cluster_radii_2d,
        cluster_num_atoms_2d,
        cluster_species_2d,
    )

    # Show as 3d structure
    view_structure(
        show_cluster(
            positions_init_2d,
            species_init_2d,
        )
    )
    return positions_init_2d, sigma2rmin, species_init_2d


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For a 3D cluster, you also need to define lists for the number of atoms per sphere with certain radii and the atom types. Here the atoms are distributed on the sphere either according to the vertices (corners) of Platonic solids with 4 atoms are distributed like the vertices of a tetrahedron, 6 -> octahedron, 8 -> cube, 12 -> isocahedron or 20 -> dodecahedron. For any other number of atoms, they are distributed with respect to a Fibonacci lattice on a sphere (e.g. https://observablehq.com/@meetamit/fibonacci-lattices)
    """)
    return


@app.cell
def _(get_cluster_3d, show_cluster, sigma2rmin, view_structure):
    # 3D Cluster setup
    cluster_radii_3d = [0.0, 0.20*sigma2rmin]
    cluster_num_atoms_3d = [1, 12]
    cluster_species_3d = [
        ['C'],
        [
            'N',
            'H', 'H', 'H', 'H', 'H',
            'S', 'S', 'S', 'S', 'S', 'S', 
        ],
        [
            'N',
            'H', 'H', 'H', 'H', 'H',
            'S', 'S', 'S', 'S', 'S', 'S', 
        ],
    ]

    # Return an ASE atom cluster with atoms on circles at different radii
    positions_init_3d, species_init_3d = get_cluster_3d(
        cluster_radii_3d,
        cluster_num_atoms_3d,
        cluster_species_3d,
    )

    # Show as 3d structure
    view_structure(
        show_cluster(
            positions_init_3d,
            species_init_3d,
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now that we have the opportunity to model an initial 2D or 3D LJ-cluster, we assign the respective initial positions, species, number of atoms and coordinates. Note, that <i> marimo </i> does not 'like' variables with the same name but in different cells, so don't hesitate to assign detailed variable names which can be easily understood.

    The potential model of a LJ cluster is defined by the sum of atom pair potentials with respect to the atom pairs distances. Here, we ask provide the next to the LJ potential <code> D1.LJ1 </code> also a repulsive Weeks-Chandler-Andersen (WCA) potential <code> D1.WCA1 </code> (the <code> 1 </code> indicates the 1-dimensionality, distances, of the potentials). Mathematically they are described by
    <math display="block" style="font-size: 1.5rem;">
        <mrow>
            <mi>V<sub>LJ</sub></mi>
            <mo>=</mo>
            4<mn>&epsilon;</mn>
        </mrow>
        <mrow>
            <mo>[</mo>
                <msup>
                    <mrow>
                        <mo>(</mo>
                        <mfrac>
                            <mn>&sigma;</mn>
                            <mi>r</mi>
                        </mfrac>
                        <mo>)</mo>
                    </mrow>
                    <mn>12</mn>
                </msup>
                -
                <msup>
                    <mrow>
                        <mo>(</mo>
                        <mfrac>
                            <mn>&sigma;</mn>
                            <mi>r</mi>
                        </mfrac>
                        <mo>)</mo>
                    </mrow>
                    <mn>6</mn>
                </msup>
            <mo>]</mo>
        </mrow>
        <mrow>
            <mo>=</mo>
            <mn>&epsilon;</mn>
        </mrow>
        <mrow>
            <mo>[</mo>
                <msup>
                    <mrow>
                        <mo>(</mo>
                        <mfrac>
                            <mi>r<sub>min</sub></mi>
                            <mi>r</mi>
                        </mfrac>
                        <mo>)</mo>
                    </mrow>
                    <mn>12</mn>
                </msup>
                -
                <msup>
                    <mrow>
                        2
                        <mo>(</mo>
                        <mfrac>
                            <mi>r<sub>min</sub></mi>
                            <mi>r</mi>
                        </mfrac>
                        <mo>)</mo>
                    </mrow>
                    <mn>6</mn>
                </msup>
            <mo>]</mo>
        </mrow>
    </math>
    and
    <math display="block" style="font-size: 1.5rem;">
        <mrow>
            <mi>V<sub>WCA</sub></mi>
            <mo>=</mo>
        </mrow>
        <mrow>
            <mo>{</mo>
            <mrow>
                <mfrac linethickness="0">
                    <mrow>
                        <mi>V<sub>LJ</sub></mi> &nbsp; for &ensp; <mi>r</mi><mo>&lt;</mo><mi>r<sub>min</sub></mi>
                    </mrow>
                    <mrow>
                        0 &nbsp; for &ensp; <mi>r</mi><mo>&ge;</mo><mi>r<sub>min</sub></mi>
                    </mrow>
                </mfrac>
            </mrow>
        </mrow>
    </math>
    with <math display="inline" style="font-size: 1.2rem;"> <mi>r</mi> </math> as the atom pair distance
    and LJ parameters <math display="inline" style="font-size: 1.2rem;"><mn>&epsilon;</mn></math> and <math display="inline" style="font-size: 1.2rem;"><mn>&sigma;</mn></math> with <math display="inline" style="font-size: 1.2rem;"> <mi>r<sub>min</sub></mi> <mo>=</mo> <mroot>
        <mi>2</mi>
        <mi>6</mi>
      </mroot>
      &middot;<mn>&sigma;</mn>
      <mo>	&asymp;</mo> 1.12<mn>&sigma;</mn>
    </math>.

    In this implementation, the potential is defined by a dictionary <code> cluster_1d_potentials </code> where the keys are a tuple of two atom species between which a 1D potential acts that is defined in the item. The item is a list where the first element is the 1D potential class (<code> D1.LJ1 </code> or <code> D1.WCA1 </code>) and the following elements are the potential parameter. The most significant definition will be the potential between the center atom C and the leaving atom N which is defined here as LJ potential with a potential depth of 10.0 kJ/mol and a &sigma; value of 0.20 nm.
    """)
    return


@app.cell
def _(D1, get_potential, np, positions_init_2d, species_init_2d):
    # Decide whether you want to simulate in 2D or 3D and store the cluster positions and species for later use
    positions_init = positions_init_2d
    species_init = species_init_2d
    natoms = positions_init.shape[0]
    ndims = positions_init.shape[1]

    np.savez(
        'lj_cluster_init.npz',
        positions=positions_init,
        species=species_init,
        natoms=natoms,
        ndims=ndims,
    )

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

    # Prepare atom pair potential but only for atom pairs with defined pair parameters.
    potential_md = get_potential(
        ndims=ndims,
        natoms=natoms,
        species=species_init,
        potential_parameters=cluster_1d_potentials,
    )

    # Test the potential function on the initial cluster
    energy_init = potential_md.potential(positions_init)
    forces_init = potential_md.force_ana(positions_init)
    print(
        f"The initial cluster energy is {energy_init:.3f} kJ/mol\n"
        + f"with a maximum force component of {np.amax(forces_init):.3f} kJ/mol/nm"
    )
    return natoms, ndims, positions_init, potential_md, species_init


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Before we start dynamics simulations, we optimize the cluster structure with repsect to its potential energy. Even though the atom masses are not required for the optimization, they can be used to mark atoms to be fixed in space.

    Physically, an atom can be fixed by increasing its mass to infinity. Technically, it is easier to set the mass of fixed atoms to zero (e.g. as done in openmm).

    <b> TASK </b>
    Optimize your cluster and, eventually, adjust your potential parameters or cluster build to achieve a suitable configuration where the center and leaving atom are still bonded and surrounded by 'solvent' atoms.
    """)
    return


@app.cell
def _(
    ndims,
    np,
    optimize_cluster,
    positions_init,
    potential_md,
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
        potential=potential_md,
        mask=mask_min,
    )

    # Show as 3d structure
    view_structure(
        show_cluster(
            positions_min,
            species_init,
        ),
        recording_tools=True,
        measuring_tool=True,
    )
    return cluster_masses, positions_min


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Check the effective potential depth between center and leaving atom.
    """)
    return


@app.cell
def _(ndims, np, plt, positions_min, potential_md, scan_distance):
    # Compute the 1D potential curve for the dissociation process using the optimized cluster
    distances = np.arange(0.15, 1.0, 0.001)
    energies = scan_distance(
        distances=distances,
        positions=positions_min,
        potential=potential_md,
        atom_i=0,
        atom_j=1,
        ndims=ndims,
    )

    # Store pair ponetial result
    np.savez(
        'unbinding_potential_md.npz',
        positions=positions_min,
        distances=distances,
        energies=energies,
    )

    # Shift energies tail to zero for pair potential
    pair_energies = energies - energies[-1]

    # Plot atom pair potential
    fig1, axs1 = plt.subplots()

    axs1.plot(
        distances,
        pair_energies,
        color='black',
        label=r'Atom Pair Potential'
    )

    axs1.set_ylim(1.1*np.min(pair_energies), -0.15*np.min(pair_energies))
    axs1.set_xlim(distances[0], 0.5)

    axs1.set_xlabel('Distance [nm]')
    axs1.set_ylabel('Potential [kJ/mol]')

    axs1.legend(loc='upper right', framealpha=1.0)

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let us start with a first simulation of the cluster. First the integrator parameters (simulation temperature, friction coefficient and time step) is defined and collected into one system instance with the variable name <code> cluster </code>.

    <b> TASK 1 </b>
    Modify the code below that the simulation terminates when the distance between the center atom C and leaving atom N becomes larger than a threshold distance and store the elapsed simulation time.

    <b> TASK 2 </b>
    Furhter extend the code to run the simulation N times to sample a set of leaving times. Best is also to save the list of leaving times in a file as a backup and for further evaluation.
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
    potential_md,
    show_cluster,
    species_init,
    system,
    view_structure,
):
    # First, the Langevin integrator parameters must be defined
    temperature = 300.0 # Kelvin, rememer <E_kin> = kB*T = 0.008314 kJ⋅mol-1⋅K-1 * T, <E_kin(300K)> = 2.5 kJ/mol
    friction = 1.0 # 1/ps
    time_step = 0.002 # ps

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

    # How many steps should the MD simulation run
    Nsteps = 100_000 # Nsteps*time_step = 200ps

    # Set cluster positions and velocities with respect to Boltzman distribution
    cluster.x = positions_min
    cluster.v = get_thermalized_velocities_nd(temperature, masses, scale_exactly=True)

    # Create trajectory position and velocity arrays and store the initial conditions
    trajectory_test_positions = np.zeros([Nsteps + 1, cluster.x.shape[0]], dtype=float)
    trajectory_test_velocities = np.zeros([Nsteps + 1, cluster.v.shape[0]], dtype=float)

    trajectory_test_positions[0, :] = cluster.x.copy()
    trajectory_test_velocities[0, :] = cluster.v.copy()

    # Run the simulation by iterating over time steps
    for _step in range(Nsteps):

        # Perform the time step by the Langevin integrator
        Dn_stochastic.BOAOB(
            cluster,
            potential_md,
            fixed=mask_md,
        )

        # Store positions and velocities
        trajectory_test_positions[_step + 1, :] = cluster.x.copy()
        trajectory_test_velocities[_step + 1, :] = cluster.v.copy()

        # Student task:

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
            print(f"Unbinding at {_step*time_step:6.3f} ps!")
            break

    # Show as 3d trajectory
    view_structure(
        [
            show_cluster(positions_i.reshape(natoms, ndims), species_init)
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
    return (
        Nsteps,
        cluster,
        mask_md,
        masses,
        md_threshold_atom_i,
        md_threshold_atom_j,
        md_threshold_distance,
        temperature,
        time_step,
    )


@app.cell
def _(
    Dn_stochastic,
    Nsteps,
    cluster,
    get_thermalized_velocities_nd,
    mask_md,
    masses,
    md_threshold_atom_i,
    md_threshold_atom_j,
    md_threshold_distance,
    natoms,
    ndims,
    np,
    positions_min,
    potential_md,
    temperature,
    time_step,
):
    # Student task:

    md_unbinding_times = []

    # How many simulations will be performed
    Nsimulation = 100

    # Run simulation n time with different initial velocity distributions
    for irun in range(Nsimulation):

        # Reset cluster positions and velocities with respect to Boltzman distribution
        cluster.x = positions_min
        cluster.v = get_thermalized_velocities_nd(temperature, masses, scale_exactly=True)

        # Create trajectory position and velocity arrays and store the initial conditions
        trajectory_md_positions = np.zeros([Nsteps + 1, cluster.x.shape[0]], dtype=float)
        trajectory_md_velocities = np.zeros([Nsteps + 1, cluster.v.shape[0]], dtype=float)

        trajectory_md_positions[0, :] = cluster.x.copy()
        trajectory_md_velocities[0, :] = cluster.v.copy()

        # Run the simulation by iterating over time steps
        sucess = False
        for _step in range(Nsteps):

            # Perform the time step by the Langevin integrator
            Dn_stochastic.BOAOB(
                cluster,
                potential_md,
                fixed=mask_md,
            )

            # Store positions and velocities
            trajectory_md_positions[_step + 1, :] = cluster.x.copy()
            trajectory_md_velocities[_step + 1, :] = cluster.v.copy()

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
                print(
                    f"({irun + 1:3d}/{Nsimulation:d}) Unbinding at {_step*time_step:6.3f} ps "
                    + f"({md_unbinding_mean:4.1f} +- {md_unbinding_rmsd:4.1f} ps)"
                )
                sucess = True
                break

        if not sucess:
            print(f"({irun + 1:d}/{Nsimulation:d}) No unbinding occured")

    # Show unbinding time
    md_unbinding_mean = np.mean(md_unbinding_times)
    md_unbinding_rmsd = np.std(md_unbinding_times)
    print(f"Mean unbinding time at {md_unbinding_mean:.3f} +- {md_unbinding_rmsd:.3f} ps")

    # Store results
    np.savetxt('unbinding_times_md.dat', md_unbinding_times)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    While the simulations are running and sampling leaving times, we can start to prepare the evaluation of the results.

    <b> TASK 1 </b>
    Starting from the list of leaving times, compute and plot the survival probability.

    <b> TASK 2 </b>
    Estimate the unbinding rate either by fitting a exponential decay function to the expontial part of your survival probability (e.g. via <code> scipy.optimize.curve_fit </code>) or by equation (2.8).
    """)
    return


@app.cell
def _(np, plt, sp):
    # Load unbinding times results
    md_unbinding_times_eval = np.loadtxt('unbinding_times_md.dat')

    # Student task:

    # First, compute a time histogram of the unbinding times
    eval_tau_range = [0.0, 50.0] #ps
    eval_tau_nbins = 50 # 1.0 ps bins
    eval_tau_edges = np.linspace(*eval_tau_range, num=(eval_tau_nbins + 1))
    eval_tau_step = eval_tau_edges[1] - eval_tau_edges[0]
    eval_tau_centers = eval_tau_edges[:-1] + eval_tau_step/2.

    md_unbinding_times_histogram, _ = np.histogram(
        md_unbinding_times_eval,
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
        eval_tau_centers > 10.0,
        eval_tau_centers < 30.0,
    )

    md_unbinding_survival_rates = (
        -der_md_unbinding_survival_propability
        / md_unbinding_survival_propability
    )

    md_unbinding_survival_rate_mean = np.mean(md_unbinding_survival_rates[mask_rate])

    # Try exponential curve fit
    def exponential_decay(t, A, k):
        return A*np.exp(-k*t)

    popt, _ = sp.optimize.curve_fit(
        exponential_decay,
        eval_tau_centers[mask_rate],
        md_unbinding_survival_propability[mask_rate],
        p0=[1.0, 1.0]
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
    taxs2.set_ylim(-.2, 0.75)

    axs2.set_xlabel('Time t [ps]')
    axs2.set_ylabel('Survival Probability')
    taxs2.set_ylabel('Time Derivative [1/ps]', color='red')

    axs2.legend(loc='upper left', framealpha=1.0)
    taxs2.legend(loc='upper right', framealpha=1.0)

    print(r'Survival rate by exponential fit: $k_{AB,1}$ = ' + f'{popt[1]:.3f} 1/ps')
    print(r'Survival rate by linear fit: $k_{AB,2}$ = ' + f'{md_unbinding_survival_rate_mean:.3f} 1/ps')

    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
