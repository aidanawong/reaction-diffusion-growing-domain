from resources import grow_func, reaction_func
from solve_pde import ReactDiffusion

def main():

    width = 1
    D, R, init = reaction_func.schnak()
    dt = 0.1
    dx = 0.01 
    T = 800
    
    r, rt = grow_func.growlog()

    steps = 70
    n_steps = round(T / (steps * dt))

    ReactDiffusion(
        D, R, r, rt, 
        ic=init, bc="neumann",
        width=width, dx=dx, dt=dt,
        steps=steps, labels=["Tulip Pigments", "Building Blocks", "Virus"], cmaps=["PuRd", "Greens", "Blues"]
    ).plot_tri("reaction-diffusion-growing-domain/figures/test.png", n_steps=n_steps, symmetry=False, radius=None, flipx=False, flipy=False)

main()