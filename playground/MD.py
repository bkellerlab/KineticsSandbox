import numpy as np
import matplotlib.pyplot as plt
import os




# Reflective boundary condition function
def reflectBC(r, v, n, D, L):
    """
    Apply reflective boundary conditions.
    
    Parameters:
    r  -- positions array
    v  -- velocities array
    L  -- system size (box dimensions)
    
    Returns:
    newr -- updated positions
    newv -- updated velocities
    """
    newv = 1.0 * v
    newr = 1.0 * r
    for i in range(n):  # Loop over all particles
        for j in range(D):  # Loop over all dimensions
            if r[i][j] < 0:  # If particle is out of bounds on the lower side
                newr[i][j] = -newr[i][j]  # Reflect position
                newv[i][j] = abs(v[i][j])  # Reverse velocity direction (positive)
            if r[i][j] > L[j]:  # If particle is out of bounds on the upper side
                newr[i][j] = 2.0 * L[j] - newr[i][j]  # Reflect position
                newv[i][j] = -abs(v[i][j])  # Reverse velocity direction (negative)
    
    return newr, newv


def dump(r, t, L, tp):
    """
    Dump particle positions for visualization.
    
    Parameters:
    r -- positions array (n, D)
    t -- current timestep
    L -- system size (box dimensions)
    """
    # Ensure the 'dumps' directory exists
    if not os.path.exists("dumps"):
        os.makedirs("dumps")
    
    # Create the filename for the current timestep
    fname = "dumps/t" + str(t) + ".dump"
    
    with open(fname, "w") as f:
        # Write the timestep information
        f.write("ITEM: TIMESTEP\n")
        f.write(str(t) + "\n")  # timestep
        
        # Write the number of atoms
        f.write("ITEM: NUMBER OF ATOMS\n")
        f.write(str(len(r)) + "\n")  # number of atoms
        
        # Write the box bounds
        f.write("ITEM: BOX BOUNDS pp pp pp\n")  # pp = periodic BCs
        f.write("0 " + str(L[0]) + "\n")
        f.write("0 " + str(L[1]) + "\n")
        f.write("0 " + str(L[2]) + "\n")
        
        # Write the atom positions
        f.write("ITEM: ATOMS id mol type x y z\n")
        for i in range(len(r)):
            f.write(f"{i+1} {i+1} {tp[i]} {r[i][0]} {r[i][1]} {r[i][2]}\n")


# Plot function to display the particle positions in 3D
def plot_particles(r, L):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each particle as a point
    ax.scatter(r[:, 0], r[:, 1], r[:, 2], c='r', marker='o')
    
    # Set plot limits to the box size
    ax.set_xlim([0, L[0]])
    ax.set_ylim([0, L[1]])
    ax.set_zlim([0, L[2]])

    ax.set_xlabel('X [nm]')
    ax.set_ylabel('Y [nm]')
    ax.set_zlabel('Z [nm]')
    
    plt.show()

def force(r, i, potential, L, cutoff=None, BC=0):
    """
    Compute the Lennard-Jones forces acting on particle i using periodic boundary conditions.

    Parameters:
    r         -- positions array (n, D)
    i         -- index of the particle for which the force is calculated
    potential -- Potential object (e.g., an instance of LennardJones)
    L         -- box dimensions array (D,)
    cutoff    -- cutoff distance beyond which interactions are ignored (optional)

    Returns:
    F_i -- force vector acting on particle i due to all other particles
    """
    n, D = r.shape

    # Calculate displacement vectors between particle i and all other particles
    drv = r - r[i]  # (n, D)
    drv = np.delete(drv, i, axis=0)  # Remove self-interaction (n-1, D)


    if BC == 0:
        # Apply minimum image convention
        drv = drv - L * np.round(drv / L)

    # Compute distances
    dr = np.linalg.norm(drv, axis=1)

    # Apply cutoff if provided
    if cutoff is not None:
        mask = dr < cutoff  # Mask out distances greater than the cutoff
        drv = drv[mask]
        dr = dr[mask]

    # Avoid division by zero or very small distances
    small_cutoff = 1e-12
    dr = np.clip(dr, small_cutoff, None)

    # Calculate the force magnitude using the Lennard-Jones potential
    F_magnitude = potential.force_ana(dr)  # (m,)

    # Normalize the displacement vectors
    unit_vectors = drv / dr[:, np.newaxis]  # (m, D)

    # Compute the force vectors
    F_vectors = F_magnitude[:, np.newaxis] * unit_vectors  # (m, D)

    # Sum the forces to get the total force on particle i
    F_i = np.sum(F_vectors, axis=0)  # (D,)

    return F_i





def updatev(r, v, dt, potential, m, L, cutoff=None, BC=0):
    """
    Update velocities using the specified potential with a cutoff.

    Parameters:
    r         -- positions array (n, D) where n is the number of particles and D is the number of dimensions
    v         -- velocities array (n, D)
    dt        -- time step
    potential -- potential object (must implement force_ana method)
    m         -- mass of particles (array of shape (n,))
    cutoff    -- cutoff distance beyond which interactions are ignored (optional)

    Returns:
    newv -- updated velocities
    a    -- acceleration (force/mass)
    """
    n, D = r.shape  # Number of particles and dimensions

    # Initialize force array
    F = np.zeros_like(r)

    # Compute forces for each particle
    for i in range(n):
        F[i] = force(r, i, potential, L, cutoff, BC)  # Call force to compute force on particle i

    # Reshape mass array to allow broadcasting
    m = m[:, np.newaxis]  # Shape becomes (n, 1) to align with (n, D)

    # Compute acceleration: a = F / m (broadcasted correctly)
    a = F / m

    # Update velocities: v(t+dt) = v(t) + a(t) * dt
    newv = v + a * dt

    return newv, a


# Update subroutine
def update(r, v, dt, n, D, L, BC=0):
    """
    Update the positions and velocities, applying the chosen boundary condition.
    
    Parameters:
    r  -- positions array
    v  -- velocities array
    dt -- time step
    L  -- system size (box dimensions)
    BC -- boundary condition flag: 0 for periodic, 1 for reflective
    
    Returns:
    newr -- updated positions
    newv -- updated velocities
    """
    # Update positions based on current velocity
    newr = r + v * dt
    
    # Initialize new velocities
    newv = 1.0 * v  # Copy of velocity array
    
    # Apply periodic boundary conditions if BC == 0
    if BC == 0:
        newr = newr % L  # Wrap particles around if they exceed boundaries
    
    # Apply reflective boundary conditions if BC == 1
    elif BC == 1:
        newr, newv = reflectBC(newr, v, n, D, L)  # Reflect at the boundaries
    
    return newr, newv


def rescaleT(v, T_target, m, R=8.314462618):  # R in J/(mol·K)
    """
    Rescale velocities to achieve target temperature.

    Parameters:
    v        -- velocities array (n, D) in nm/ps
    T_target -- the target temperature (in Kelvin)
    m        -- mass of particles in g/mol (array of shape (n,))
    R        -- Universal gas constant in J/(mol·K)

    Returns:
    v_new -- rescaled velocities to achieve the target temperature
    """
    # Convert R to kJ/(mol·K)
    R_kJ = R / 1000  # R_kJ is in kJ/(mol·K)

    # Compute the number of particles
    n = v.shape[0]

    # Compute the kinetic energy in kJ/mol
    KE = 0.5 * np.sum(m[:, np.newaxis] * v**2)  # Units: (g/mol)*(nm/ps)^2

    # Since (kJ/g) = (nm/ps)^2, and mass is in g/mol, KE is in kJ/mol

    # Compute the current temperature
    T_now = (2 * KE) / (3 * n * R_kJ)

    # Rescaling factor
    lambda_factor = np.sqrt(T_target / T_now)

    # Rescale velocities
    v_new = v * lambda_factor

    return v_new, T_now


def calculate_number_of_particles(temperature_K, pressure_Pa, box_length_nm):
    """
    Calculate the number of particles N to include in a simulation box using the ideal gas law.

    Parameters:
    - temperature_K (float): Temperature in Kelvin (K).
    - pressure_Pa (float): Pressure in Pascals (Pa).
    - box_length_nm (float or list/tuple): Length of the simulation box in nanometers (nm). 
      If the box is cubic, provide a single float. For a rectangular box, provide a list or tuple of three floats.

    Returns:
    - N (int): Number of particles to include in the simulation box.
    """

    # Constants
    R = 8.31446261815324  # Ideal gas constant in J/(mol·K)
    N_A = 6.02214076e23   # Avogadro's number in particles/mol

    # Convert box length(s) from nm to m
    if isinstance(box_length_nm, (list, tuple, np.ndarray)):
        # For non-cubic boxes
        box_lengths_m = np.array(box_length_nm) * 1e-9  # Convert each dimension to meters
        volume_m3 = np.prod(box_lengths_m)
    else:
        # For cubic boxes
        box_length_m = box_length_nm * 1e-9  # Convert to meters
        volume_m3 = box_length_m ** 3

    # Calculate the number of moles using the ideal gas law: n = pV / (RT)
    n_moles = (pressure_Pa * volume_m3) / (R * temperature_K)

    # Calculate the number of particles: N = n * N_A
    N_particles = n_moles * N_A

    # Round to the nearest whole number, since we can't have a fraction of a particle
    N = int(round(N_particles))

    return N



## old 
#def force(r, i, potential, cutoff=None):
#    """
#    Compute the Lennard-Jones forces acting on particle i #using a vectorized approach.
#    
#    Parameters:
#    r            -- positions array (n, D) where n is the #number of particles and D is the number of dimensions
#    i            -- index of the particle for which the force #is calculated
#    potential    -- Potential object (an instance of in #example LennardJones)
#    cutoff       -- cutoff distance beyond which interactions #are ignored (optional)
#    
#    Returns:
#    F_i -- force vector acting on particle i due to all other #particles
#    """
#    # Calculate distance vectors between particle i and all #other particles
#    drv = r - r[i]  # Vector of differences
#    drv = np.delete(drv, i, axis=0)  # Remove self-interaction
#
#    # Compute Euclidean distances
#    dr = np.linalg.norm(drv, axis=1)
#
#    # Apply cutoff if provided
#    if cutoff is not None:
#        mask = dr < cutoff  # Mask out distances greater than #the cutoff
#        drv = drv[mask]
#        dr = dr[mask]
#
#    # Avoid division by zero or very small distances
#    small_cutoff = 1e-12
#    dr = np.clip(dr, small_cutoff, None)
#
#    # Calculate the force using the Lennard-Jones potential #for the distance array
#    F_magnitude = potential.force_ana(dr)  # Vectorized force #calculation
#
#    # Normalize the distance vectors and multiply by the #force magnitudes
#    unit_vectors = drv / dr[:, np.newaxis]  # Create unit #vectors
#    F_i = np.sum(F_magnitude[:, np.newaxis] * unit_vectors, #axis=0)  # Sum of force vectors
#
#    return F_i