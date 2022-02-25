# PyBAND
Python version of John Newman's FORTRAN BAND(J)

Uses Numba for improved computational performance
Comparison of ToySteadyStateDiffusion.py shows that the use of NumBa jit improves the computational speed by 20-30x, which is quite good. 

The next target is to use JAX instead of NumBa to jit the code and take advantage of automatic differentiation.
