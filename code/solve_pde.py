"""
Reaction Diffusion Equation on Growing 1D Domain
Numerically calculates a system of reaction diffusion equation over time
Author: Aidan Wong
Date: May 20, 2024
"""

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
                 cmaps=["Tulip", "Greens", "Blues", "Oranges", "RdPu", "GnBu"]):
        
        # Array of diffusion coefficients
        self.D = np.array(D)

        # Array of reaction functions
        self.R = R

        # Spatial parameters
        self.r = r # normalized domain growth function
        self.rt = rt # first order derivative of growth function
        self.L = lambda t: width * r(t) # function of domain growth over time
        self.N = round(width / dx) # number of points in space
        self.Xarray = np.linspace(0, 1, self.N) # initial spatial array
        self.width = width # initial width
        
        # Temporal parameters
        self.dt = dt 
        self.dx = dx / width # Scale to a width of 1
        self.steps = steps # number of steps between saving data

        # Initial and boundary conditions respectively
        self.ic = ic
        self.bc = bc

        # Set up time discretization constants
        self.P = lambda t: self.D * dt / (2 * (self.L(t) * self.dx)**2)
        self.Q = lambda t: rt(t) * dt / (2 * r(t))

        # Graphing parameters
        self.labels = labels
        self.cmaps = cmaps        
        
    def initialise(self):
        """
        Initializes the PDE by handling errors, and finding the number of variables
        """
        assert (self.bc == "dirichlet" or self.bc == "neumann"), "Sorry, Please input \"dirichlet\" or \"neumann\".\nNo Robin conditions yet ;-)"
        
        error_tester = [np.size(self.D), np.size(self.R), np.size(self.ic(1))]
        assert all(size == error_tester[0] for size in error_tester), "Sorry, please check the number of variables."
        
        # Number of equations/variables in the system of PDE's
        self.Nsys = np.size(self.D)

        # Initializes time and variables
        self.t = 0
        self.U = self.ic(self.N)
        
    def update(self):
        """
        Run the equation for a user-specified number of steps
        """
        for _ in range(self.steps):
            self._update()
            self.t += self.dt
            Lnew = self.L(self.t) 
            self.Xarray = np.linspace(0, Lnew, self.N)

    def _update(self):
        """
        Advance the numerics of the PDE by one time increment
        """

        # Unpack variables
        U, t, dt, P, Q, R = self.U, self.t, self.dt, self.P, self.Q, self.R

        # Empty current variables
        self.U = np.zeros((self.Nsys, self.N))

        # Calculate current and new time discretization constants
        PP = np.array([P(t), P(t + dt)])
        QQ = [Q(t), Q(t + dt)]

        for i in range(self.Nsys):
            self.U[i] += self.run_pde(self.bc, dt, U[i], R[i](U), PP[:,i], QQ)

    def TDMAsolver(self, a, b, c, d):
        """
        Solves the tri-diagonal matrix in the form of
        |b1  c1               |   |x1|     |d1|
        |a2  b2  c2           |   |x2|     |d2|
        |    a3  b3  c3       | * |x3|  =  |d3|
        |        .    .       |   |. |     |. |
        |           .   .   . |   |. |     |. |
        |               an  bn|   |xn|     |dn|

        Credit to Theo Christiaanse
        https://gist.github.com/TheoChristiaanse/d168b7e57dd30342a81aa1dc4eb3e469?permalink_comment_id=2225268
        Inputs: 
            a (array, 1D) - first diagonal
            b (array, 1D) - second/main diagonal
            c (array, 1D) - third diagonal
            d (array, 1D) - right hand side
        Returns:
            xc (array, 1D) - solution
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
        """
        Assembles the Crank Nicolson tri diagonal matrix and solves it
        Inputs: 
            bc (str) - boundary condition, either 'dirichlet' or 'neumann'
            dt (float) - time increment
            u (array, 1D) - values of the variable over space
            Ru (list) - contains the functions of the reaction-diffusion equation
            P (float) - discretization constant
            Q (float) - discretization constant
        """
        n = len(u)

        # Set up right hand side of tri-diagonal matrix (TDM) equation
        Pold, Pnew = P
        Qold, Qnew = Q 
        coef1 = np.full(n-1, -Pnew)
        coef2 = np.full(n, 1 + 2 * Pnew + Qnew)
        coef3 = np.full(n-1, -Pnew)
        rhs = Pold * np.roll(u, -1) + (1 - 2 * Pold - Qold) * u + Pold * np.roll(u, 1) + dt * Ru
        rhs_bc = lambda i, j: (1 - 2 * Pold - Qold) * u[i] + 2 * Pold * u[j] + dt * (Ru[i])
        
        # TDM changes depending on boundary conditions
        if bc == "dirichlet":
            # Forces edges to become zero
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
        """
        Run the complete PDE system for the specified amount of time.
        Returns: 
            u_mat (array, 2D) - contains recorded values of each PDE variable
            x_mat (array, 2D) - contains recorded values of the domain growth
            t (array, 1D) - contains recorded values of time
        """
        # Initialize matrices
        N, Nsys = self.N, self.Nsys # spatial resolution and number of variables
        tf = self.dt * self.steps * n_steps # final time
        u_mat = np.zeros((n_steps, Nsys, N))
        x_mat = np.zeros((n_steps, N))
        t = np.linspace(0, tf, n_steps)

        for _ in progressbar(range(n_steps), "Calculating: "):
            self.update()
            u_mat[_] += self.U
            x_mat[_] += self.Xarray
        
        return u_mat, x_mat, t