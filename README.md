# PyBAND
Python version of John Newman's FORTRAN BAND(J)

Uses Numba for improved computational performance
Comparison of ToySteadyStateDiffusion.py shows that the use of NumBa jit improves the computational speed by 20-30x, which is quite good. 

The next target is to use JAX instead of NumBa to jit the code and take advantage of automatic differentiation.

```python
@nb.jit(nopython=True)
def unsteady(initial_conditions, number_time_steps, auto_fill, ABDGXY, BAND_NEWMAN, calc_delC):
    c_prev = initial_conditions
    yield c_prev

    for it in range(number_time_steps):
        rj, fE, fW, dE, dW, smG = auto_fill(c_prev, beta, delV, delT)   # automatic differentiation only used here
        A, B, D, G, X, Y        = ABDGXY(alphaW, alphaE, betaW, betaE, rj, fE, fW, dE, dW, smG)
        E, xi, x                = BAND_NEWMAN(A, B, D, G, X, Y)         # calculate E, ξ, x
        delC                    = calc_delC(NJ, E, xi, x)               # calculate Δc

        c_prev = c_prev + delC

        yield c_prev
```
This works very well and each subfuntion listed here as well as `MATINV` has also been jitted to achieve the 20x speed up.
One issue that remains is that the first compilation takes longer because of the additional NumBa / JIT overhead. We can get around this by using Ahead-of-Time (AoT) compilation instead of Just-in-Time (JIT) compilation. However, with number, only functions that have a single output can be compiled ahead of time. In the current structure, that limits us to `calc_delC` and `MATINV`, but with some restructing, `ABDGXY` could be done through AoT `calc_A`, `calc_B`, `calc_D`, `calc_G`, `calc_X`, `calc_Y`. In addition, perhaps with some cleverly designed matrices, `auto_fill` and `BAND` could also be compiled AoT. I hesitate to do that though because "cleverly" designing matrices decreases the readability of the code. 

`auto_fill(c_prev, beta, delV, delT)` needs to be modified to `auto_fill(c_prev, beta, delV, delT, gov_eqns)` where `gov_eqns` will somehow encapsulate the flux, accumulation, reaction, and boundary conditions
