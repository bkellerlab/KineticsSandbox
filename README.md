# GirsanovReweighting_Benchmark

## Package potentials

This package contains modules for various test potentials. Each test potential gets its own module. The module implements the functions

- V: potential
- F: analytical force
- F_numerical: numerical force via finite difference
- p: unnormalized Boltzmann distribution
- Q: partition function

For now everything should be implemented in  Cartesian coordinates. For the 1-dimensional and 2-dimensional potentials, I have think how to handle circular and spherical coordinates.

The naming convention for the potentials is

- D1_potential: 1-dimensional potentials
- D2_potential: 2-dimensional potentials
- D3_potential: 3-dimensional potentials

(Python forbids module names that start with a digit.)