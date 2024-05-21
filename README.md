# reaction-diffusion-growing-domain
 Solves 1D reaction diffusion equations on a growing domain

## How to Use

1. Enter your reaction functions inside `resources/reaction_func.py` following the formats provided
2. Include your initial conditions with your reaction functions
3. Enter or edit your growth functions inside `resources/grow_func.py`. This function should be normalized such that it starts at 1. Choose `growno()` if you do not want any domain growth.
4. Also, add the derivative of your growth function.
4. Go to `main.py`, set desired values, and run!
```
def main():

    width = 1
    D, R, init = reaction_func.tbv_model_dim()
    dt = 0.1
    dx = 0.01 
    T = 800
    
    r, rt, radius = grow_func.growforward(width=width, time_mult=T/ORIGINAL_PETAL_LENGTH)

    steps = 10
    n_steps = round(T / (steps * dt))

    ReactDiffusion(
        D, R, r, rt, 
        ic=init, bc="neumann",
        width=width, dx=dx, dt=dt,
        steps=steps, labels=["Tulip Pigments", "Building Blocks", "Virus"], cmaps=["PuRd", "Greens", "Blues"]
    ).plot_tri("reaction-diffusion-growing-domain/figures/testo.png", 
               n_steps=n_steps, symmetry=True, radius=radius, flipx=False, flipy=False)

main()
```
- `width` is the initial domain width
- `D` is an array of diffusion coefficients
- `R` is an array of reaction functions for the PDE
- `init` is a function containing the initial conditions
- `dt` and `dx` are the time and space increments respectively
- `T` is the final time
- `r` and `rt` are your normalized growth function and its derivative respectively
- If you want to curve your graph to a petal-like shape or otherwise, `radius` is a function that defines the radius of curvature as a function of time
- `steps` is the number of calculation steps between recording data for plotting it (i.e. the greater the steps, the lower the resolution.)
- `n_steps` is the number of times the simulation runs for the number of `steps` (i.e. the total number of calculations will be `n_steps * steps`)
- Also, the boundary conditions `bc` can be either `neumann` for Neumann conditions or `dirichlet` for Dirichlet conditions.
- Change `labels` for the labels for each variable
- Change `cmaps` for different colour maps.
- If True, `flipx` makes the x-axis run from maximum to minimum as one goes from left to right.
- If True, `flipy` makes the x-axis denote space and the y-axis denote time.
- If True, `symmetry` centers the plot around the line where space equals zero. It needs to be True before any curvature can be applied!