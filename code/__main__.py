"""
Reaction Diffusion Equation on Growing 1D Domain
User interface for calculating PDEs
Author: Aidan Wong
Date: May 20, 2024
"""

from resources import grow_func, reaction_func
from resources.grow_func import ORIGINAL_PETAL_LENGTH
from solve_pde import ReactDiffusion

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