import numpy as np
from tutils import BaseStateSystem
from resources.reaction_func import initalise_bump
from resources.progress_bar import progressbar

class ReactDiffusion(BaseStateSystem):
    def __init__(self, D, R, r, rt,
                 ic=initalise_bump, bc="neumann",
                 width=1000, dx=1, 
                 dt=0.1, steps=1,
                 labels=["Tulip Pigments", "Building Blocks", "Virus"], 
                 colors=["r", "g", "b", "c", "m", "y", "k"], cmaps = ["Tulip", "Greens", "Blues", "Oranges", "RdPu", "GnBu"]):
            
        self.D = np.array(D)
        self.R = R

        self.r = r
        self.rt = rt
        self.L = lambda t: width * r(t)
        self.N = round(width / dx)
        self.Xarray = np.linspace(0, 1, self.N)
        self.width = width
        
        self.dt = dt
        self.dx = dx / width # Scale to a width of 1
        self.steps = steps

        self.ic = ic
        self.bc = bc

        self.short_arr = np.ones(self.N - 1)
        self.long_arr = np.ones(self.N)
        self.P = lambda t: self.D * dt / (2 * (self.L(t) * self.dx)**2)
        self.Q = lambda t: rt(t) * dt / (2 * r(t))

        self.labels = labels
        self.colors = colors
        self.cmaps = cmaps        
        
    def initialise(self):
        assert (self.bc == "dirichlet" or self.bc == "neumann"), "Sorry, Please input \"dirichlet\" or \"neumann\".\nNo Robin conditions yet ;-)"
        
        error_tester = [np.size(self.D), np.size(self.R), np.size(self.ic(1))]
        assert all(size == error_tester[0] for size in error_tester), "Sorry, please check the number of variables."
        
        self.Nsys = np.size(self.D)

        self.t = 0
        self.U = self.ic(self.N)
        
    def update(self):
        for _ in range(self.steps):
            self._update()
            self.t += self.dt
            Lnew = self.L(self.t) 
            self.Xarray = np.linspace(0, Lnew, self.N)

    def _update(self):
        U, t, dt, P, Q, R = self.U, self.t, self.dt, self.P, self.Q, self.R
        self.U = np.zeros((self.Nsys, self.N))
        PP = np.array([P(t), P(t + dt)])
        QQ = [Q(t), Q(t + dt)]
        for i in range(self.Nsys):
            self.U[i] += self.run_pde(self.bc, dt, U[i], R[i](U), PP[:,i], QQ)

    def TDMAsolver(self, a, b, c, d):
        """
        Credit to Theo Christiaanse
        https://gist.github.com/TheoChristiaanse/d168b7e57dd30342a81aa1dc4eb3e469?permalink_comment_id=2225268
        """
        # Tri Diagonal Matrix Algorithm (a.k.a Thomas algorithm) solver
        nf = len(d) # number of equations
        ac, bc, cc, dc = (a, b, c, d) # copy arrays
        for it in range(1, nf):
            mc = ac[it-1]/bc[it-1]
            bc[it] = bc[it] - mc*cc[it-1] 
            dc[it] = dc[it] - mc*dc[it-1]
                    
        xc = bc
        xc[-1] = dc[-1]/bc[-1]

        for il in range(nf-2, -1, -1):
            xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]

        return xc
            
    def run_pde(self, bc, dt, u, Ru, P, Q):
        # Makes Crank Nicolson tri diagonal matrix
        n = len(u)
        Pold, Pnew = P
        Qold, Qnew = Q 
        coef1 = np.full(n-1, -Pnew)
        coef2 = np.full(n, 1 + 2 * Pnew + Qnew)
        coef3 = np.full(n-1, -Pnew)
        rhs = Pold * np.roll(u, -1) + (1 - 2 * Pold - Qold) * u + Pold * np.roll(u, 1) + dt * Ru
        rhs_bc = lambda i, j: (1 - 2 * Pold - Qold) * u[i] + 2 * Pold * u[j] + dt * (Ru[i])
        
        if bc == "dirichlet":
            coef1.put(-1, 0)
            coef2.put([0, -1], 1)
            coef3.put(0, 0)
            rhs.put([0, -1], 0)
        elif bc == "neumann":
            coef1.put(-1, -2*Pnew)
            coef3.put(0, -2*Pnew)
            rhs.put([0, -1], [rhs_bc(0,1), rhs_bc(-1,-2)])

        return self.TDMAsolver(coef1, coef2, coef3, rhs)
        
    def run_and_retrieve(self, n_steps):
        N, Nsys = self.N, self.Nsys
        tf = self.dt * self.steps * n_steps
        u_mat = np.zeros((n_steps, Nsys, N))
        x_mat = np.zeros((n_steps, N))
        t = np.linspace(0, tf, n_steps)

        for _ in progressbar(range(n_steps), "Calculating: "):
            self.update()
            u_mat[_] += self.U
            x_mat[_] += self.Xarray
        
        return u_mat, x_mat, t