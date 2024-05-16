from resources import grow_func, reaction_func
from solve_pde import ReactDiffusion

def main():

    width = 4.8019
    D, R, init = reaction_func.tbv_model_dim()
    dt = 0.1
    dx = 0.01 
    T = 840
    
    r, rt, radius = grow_func.growbackward(width=width, time_mult=T / 6.35)

    steps = 70
    n_steps = round(T / (steps * dt))

    ReactDiffusion(
        D, R, r, rt, 
        ic=init, bc="neumann",
        width=width, dx=dx, dt=dt,
        steps=steps, labels=["Tulip Pigments", "Building Blocks", "Virus"], cmaps=["PuRd", "Greens", "Blues"]
    ).plot_tri("figures/test.png", n_steps=n_steps, symmetry=True, radius=radius, flipx=True, flipy=False)

main()