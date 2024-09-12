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
def plot_particles(r, LL):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each particle as a point
    ax.scatter(r[:, 0], r[:, 1], r[:, 2], c='r', marker='o')
    
    # Set plot limits to the box size
    ax.set_xlim([0, LL])
    ax.set_ylim([0, LL])
    ax.set_zlim([0, LL])

    #ax.set_xlim([LL/3, LL * 2/3])
    #ax.set_ylim([LL/3, LL * 2/3])
    #ax.set_zlim([LL/3, LL * 2/3])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()


    import numpy as np


def force(r, i, potential, cutoff=None):
    """
    Compute the Lennard-Jones forces acting on particle i using a vectorized approach.
    
    Parameters:
    r            -- positions array (n, D) where n is the number of particles and D is the number of dimensions
    i            -- index of the particle for which the force is calculated
    potential    -- Potential object (an instance of in example LennardJones)
    cutoff       -- cutoff distance beyond which interactions are ignored (optional)
    
    Returns:
    F_i -- force vector acting on particle i due to all other particles
    """
    # Calculate distance vectors between particle i and all other particles
    drv = r - r[i]  # Vector of differences
    drv = np.delete(drv, i, axis=0)  # Remove self-interaction

    # Compute Euclidean distances
    dr = np.linalg.norm(drv, axis=1)

    # Apply cutoff if provided
    if cutoff is not None:
        mask = dr < cutoff  # Mask out distances greater than the cutoff
        drv = drv[mask]
        dr = dr[mask]

    # Avoid division by zero or very small distances
    small_cutoff = 1e-12
    dr = np.clip(dr, small_cutoff, None)

    # Calculate the force using the Lennard-Jones potential for the distance array
    F_magnitude = potential.force_ana(dr)  # Vectorized force calculation

    # Normalize the distance vectors and multiply by the force magnitudes
    unit_vectors = drv / dr[:, np.newaxis]  # Create unit vectors
    F_i = np.sum(F_magnitude[:, np.newaxis] * unit_vectors, axis=0)  # Sum of force vectors

    return F_i


def updatev(r, v, dt, potential, m, cutoff=None):
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
        F[i] = force(r, i, potential, cutoff)  # Call force to compute force on particle i

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



def rescaleT(v, T_target, m, kb=1.0):
    """
    Rescale velocities to achieve the target temperature.
    
    Parameters:
    v        -- velocities array (n, D), where n is the number of particles and D is the number of dimensions
    T_target -- the target temperature
    m        -- mass of particles (array of shape (n,))
    kb       -- Boltzmann constant (default is 1 for reduced units)
    
    Returns:
    vnew -- rescaled velocities to achieve the target temperature
    """
    # Ensure m is an array of masses for all particles
    m = np.asarray(m)
    
    # Compute the total kinetic energy of the system
    # Kinetic energy for each particle is 1/2 m_i v_i^2, and we sum it over all particles
    KE = 0.5 * np.sum(m[:, np.newaxis] * v**2)  # Total kinetic energy: sum(1/2 m v^2)
    
    # Compute the current temperature
    T_now = (2.0 / (3 * len(v) * kb)) * KE  # T = (2/3Nk_B) * KE
    
    # Rescaling factor
    lam = np.sqrt(T_target / T_now)
    
    # Update velocities: we rescale the velocity of each particle
    v_new = lam * v
    
    return v_new



# Old Code Snippets:
# ------------------

#def LJpot(r, i, sig, eps):
#    """
#    Calculate the Lennard-Jones potential for particle i with all other particles.
#
#    Parameters:
#    r    -- positions array (n, D) where n is the number of particles and D is the number of dimensions
#    i    -- index of the particle for which the potential is calculated
#    sig  -- sigma parameter for the Lennard-Jones potential (distance at which the potential is zero)
#    eps  -- epsilon parameter for the Lennard-Jones potential (depth of the potential well)
#
#    Returns:
#    LJP -- Lennard-Jones potential for particle i
#    """
#    drv = r - r[i]  # Calculate distances between particle i and all others in each dimension
#    drv = np.delete(drv, i, axis=0)  # Remove the interaction with itself (no self-LJ interactions)
#
#    # Compute absolute distances (Euclidean norm)
#    dr = np.linalg.norm(drv, axis=1)
#
#    # Apply the Lennard-Jones formula
#    r6 = (sig / dr)**6
#    r12 = r6**2
#    LJP = 4.0 * eps * np.sum(r12 - r6)  # Sum contributions from all other particles
#
#    return LJP

#def dLJp(r, i, sig, eps, cutoff=None):
#    """
#    Calculate the force due to the Lennard-Jones potential on particle i.
#    
#    Parameters:
#    r       -- positions array (n, D) where n is the number of particles and D is the number of dimensions
#    i       -- index of the particle for which the force is calculated
#    sig     -- sigma parameter for the Lennard-Jones potential
#    eps     -- epsilon parameter for the Lennard-Jones potential
#    cutoff  -- cutoff distance beyond which interactions are ignored (optional)
#    
#    Returns:
#    dLJP -- force vector on particle i due to all other particles
#    """
#    drv = r - r[i]  # Calculate distance vectors between particle i and all others
#    drv = np.delete(drv, i, axis=0)  # Remove self-interaction
#
#    # Compute absolute distances (Euclidean norm)
#    dr = np.linalg.norm(drv, axis=1)
#    
#    # Apply cutoff if provided
#    if cutoff is not None:
#        mask = dr < cutoff  # Only consider distances within the cutoff
#        drv = drv[mask]
#        dr = dr[mask]
#    
#    # Avoid division by zero (or very small distances)
#    small_cutoff = 1e-12
#    dr = np.clip(dr, small_cutoff, None)
#
#    # Calculate r^(-8) and r^(-14) terms for the Lennard-Jones force
#    r8 = (sig / dr)**8
#    r14 = 2.0 * (sig / dr)**14
#
#    # Force magnitude calculation
#    r814 = r14 - r8
#    
#    # Multiply distance vectors by the force magnitudes
#    r814v = drv * r814[:, np.newaxis]  # Broadcasting for element-wise multiplication
#
#    # Sum all forces acting on particle i
#    dLJP = 24.0 * eps * np.sum(r814v, axis=0)
#    
#    return dLJP