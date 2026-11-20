import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## AlgoSB 2026: Learning Protein Dynamics
    # Dynamic Reweighting for Unbinding Processes

    ### (2) Biased Potential

    In this part, the original potential gets biased and used to yield unbinding times which gets reweigthed to estimate the unbinding times or rates for the original potential done previously.
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
    From the previous simulation, we now know the unbinding rate for the chosen cluster and potential parameters. In this idealized case, the unbiased simulation should have progresses fast enough to get a good rate estimation at a decent time. However, higher unbinding rates should also require shorter dynamics runs and, thus, should be quicker to compute.

    Here, the previous initial cluster setup is loaded (<code> cluster_data = np.load('lj_cluster_init.npz') </code>) with the same (adjust if needed) potential definition but also an additional bias potential of a 1D LJ potential between the central C and leaving N atom. In case of the same &sigma; parameter value, the sum of two LJ potentials between the same atoms yields one LJ potential with an effective &epsilon; parameter value formed by the sum of both.
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
            +4<mn>&epsilon;'</mn>
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
            4(<mn>&epsilon; + &epsilon;'</mn>)
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
    </math>
    Thus, by assigning a negative &epsilon; value in the bias potential, the binding strength between C and N is lowered.

    <b> TASK </b> Modify the cluster potential by a bias to reduce the binding strength between the C and N atom significantly (e.g. half or less than the original potential). The effective atom pair potential will be plotted below after a initial structure minimization and potential scan along the C-N distance.
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
        ('C', 'N'): [D1.LJ1, 15.0, 0.20],
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

    # Define atom pair bias properties in terms of:
    # [pair potential bias ('LJ1' or 'WCA1'), epsilon (kJ/mol), sigma (nm)]
    # V_LJ = V_LJ_pot + V_LJ_bias
    #      = 4*epsilon_pot*((sigma/d)**12 - (sigma/d)**6) + 4*epsilon_bias*((sigma/d)**12 - (sigma/d)**6)
    #      = 4*(epsilon_pot + epsilon_bias)*((sigma/d)**12 - (sigma/d)**6)
    cluster_1d_bias = {
        # 'C' to 'N'
        ('C', 'N'):   [D1.LJ1, -10.0, 0.20],
        # or by atom indices
        # (0, 1):   [D1.LJ1, -10.0, 0.20],
    }

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
    return ndims, positions_init, potential_biased, species_init


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


@app.cell
def _(ndims, np, plt, positions_min, potential_biased, scan_distance):
    # Load unbiased potential scan results
    results_md = np.load('unbinding_potential_md.npz')
    distances_md = results_md['distances']
    energies_md = results_md['energies']

    # Compute the 1D potential curve for the dissociation process using the optimized cluster
    distances = np.arange(0.15, 1.0, 0.001)
    energies = scan_distance(
        distances=distances,
        positions=positions_min,
        potential=potential_biased,
        atom_i=0,
        atom_j=1,
        ndims=ndims,
    )

    # Shift energies tail to zero for pair potential
    pair_energies = energies - energies[-1]
    pair_energies_md = energies_md - energies_md[-1]

    # Plot atom pair potential
    fig1, axs1 = plt.subplots()

    axs1.plot(
        distances_md,
        pair_energies_md,
        color='black',
        label=r'Original Atom Pair Potential'
    )
    axs1.plot(
        distances,
        pair_energies,
        color='red',
        label=r'Biased Atom Pair Potential'
    )

    min_pair_energy = np.min([np.min(pair_energies_md), np.min(pair_energies)])
    axs1.set_ylim(1.1*min_pair_energy, -0.25*min_pair_energy)
    axs1.set_xlim(distances[0], 0.5)

    axs1.set_xlabel('Distance [nm]')
    axs1.set_ylabel('Potential [kJ/mol]')

    axs1.legend(loc='upper right', framealpha=1.0)

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <b> TASK </b> Same as with the unbiased simulations, iterate over a number of unbinding simulations with different intial atom velocities until the C-N distance increases up to a certain threshold distance and store the unbinding times and average bias potential for the computation of the accelaration factor &alpha;. While the simulations are running, prepare code to reweight the unbinding times or rates.
    """)
    return


@app.cell
def _(
    Dn_stochastic,
    cluster_masses,
    get_thermalized_velocities_nd,
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

    # How many steps should the MD simulation run
    Nsteps = 100_000 # Nsteps*time_step = 100ps

    # Set cluster positions and velocities with respect to Boltzman distribution
    cluster.x = positions_min
    cluster.v = get_thermalized_velocities_nd(temperature, masses, scale_exactly=True)

    # Create trajectory position and velocity arrays and store the initial conditions
    trajectory_test_positions = np.zeros([Nsteps + 1, cluster.x.shape[0]], dtype=float)
    trajectory_test_velocities = np.zeros([Nsteps + 1, cluster.v.shape[0]], dtype=float)
    trajectory_test_bias = np.zeros(Nsteps + 1, dtype=float)

    trajectory_test_positions[0, :] = cluster.x.copy()
    trajectory_test_velocities[0, :] = cluster.v.copy()
    trajectory_test_bias[0] = potential_biased.potential(
        cluster.x,
        verbose=True,
    )['bias'] # dictionary with items for 'original', 'bias', 'effective' (= 'original' + 'bias')

    # Run the simulation by iterating over time steps
    for _step in range(Nsteps):

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
def _(md_unbinding_bias, md_unbinding_times, np):
    # Store results
    np.savetxt(
        'unbinding_times_biased.dat',
        np.vstack((md_unbinding_times, md_unbinding_bias)).T
    )
    return


@app.cell
def _(np):
    # Load unbinding times and bias results
    md_unbinding_times_eval, md_unbinding_bias_eval = np.loadtxt('unbinding_times_biased.dat').T
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
