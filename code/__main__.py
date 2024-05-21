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
    D, R, init = reaction_func.schnak()
    dt = 0.1
    dx = 0.005 
    T = 4000
    
    r, rt = grow_func.growexp()

    steps = 50
    n_steps = round(T / (steps * dt))

    ReactDiffusion(
        D, R, r, rt, 
        ic=init, bc="neumann",
        width=width, dx=dx, dt=dt,
        steps=steps, labels=[r"$u$", r"$v$"], cmaps=["YlOrRd", "BuPu"]# ["PuRd", "Greens", "Blues"]
    ).plot_tri("reaction-diffusion-growing-domain/figures/test.png", 
               n_steps=n_steps, symmetry=False, radius=None, flipx=False, flipy=False)

main()