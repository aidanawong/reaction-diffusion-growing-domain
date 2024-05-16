"""
Utility functions from Barr's blog post
"""
from types import FunctionType, NoneType

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.axes_grid1 import make_axes_locatable

import resources.tulip_cmap

class BaseStateSystem:
    """
    Base object for "State System".

    We are going to repeatedly visualise systems which are Markovian:
    the have a "state", the state evolves in discrete steps, and the next
    state only depends on the previous state.

    To make things simple, I'm going to use this class as an interface.
    """
    def __init__(self):
        raise NotImplementedError()

    def initialise(self):
        raise NotImplementedError()
    
    def initialise_cmap_figure(self):
        # fig = plt.figure(figsize=plt.figaspect(0.6/self.Nsys)) # original
        fig = plt.figure(figsize=plt.figaspect(0.63/self.Nsys))
        return fig
    
    def initialise_cmap_figure_flipy(self):
        fig = plt.figure(figsize=plt.figaspect(1/(0.875*self.Nsys)))
        return fig

    def update(self):
        raise NotImplementedError()
    
    def run_and_retrieve(self):
        raise NotImplementedError()
    
    def make_labels(self, ax, surf, idx, flipy):
        if flipy:
            ax.set_xlabel("Space", fontsize=16)
            ax.set_ylabel("Time", fontsize=16)
        else: 
            ax.set_xlabel("Time", fontsize=16)
            ax.set_ylabel("Space", fontsize=16)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(surf, cax=cax) # aspect=8 fraction=0.046, pad=0.04
        cbar.set_label(self.labels[idx], fontsize=16)

    def plot_surf(self, filename, n_steps):
        self.initialise()
        fig = self.initialise_cmap_figure()

        u_mat, x_mat, tarr = self.run_and_retrieve(n_steps)

        xx, tt = np.meshgrid(x_mat[0], tarr)
        
        for idx in range(self.Nsys):
            uu = u_mat[:,idx,:]
            ax = fig.add_subplot(1, self.Nsys, idx + 1, projection='3d')
            surf = ax.plot_surface(tt, xx, uu, cmap=self.cmaps[idx])
            self.make_labels(ax, surf, idx)

        plt.tight_layout()
        plt.show()

    def plot_tri(self, filename, n_steps, symmetry=False, scaling="auto", radius=None, flipx=False, flipy=False):
        self.initialise()
        if flipx: 
            ratio = 1.31447
            init_width = 0.526
        else:
            ratio = 12
            init_width = 4.8019
            
        if flipy:
            fig = self.initialise_cmap_figure_flipy()
        else:
            fig = self.initialise_cmap_figure()

        u_mat, x_mat, tarr = self.run_and_retrieve(n_steps)
        tt = np.meshgrid(np.arange(self.N), tarr)[1]

        def warp_circle(X, y, r, B, C, flipx=False):
            N = np.shape(y)[0]
            L, r = np.ptp(y, axis=1).reshape(N,1), r.reshape(N, 1)
            theta_arr = 0.5 * L / r * np.linspace(-1, 1, np.shape(y)[1])
            if flipx == True:
                x = -C / B * r * np.cos(theta_arr)
                xx = x + (X - np.min(x, axis=1)).reshape(N, 1)
            else:
                x = C / B * r * np.cos(theta_arr)
                xx = x + (X - np.max(x, axis=1)).reshape(N, 1)
            yy = r * np.sin(theta_arr)
            return xx, yy
        
        def long_edges(x, y, triangles, radio=0.1):
            d = np.zeros_like(triangles.T, dtype=float)
            for i in range(3):
                d[i] = d[i] + np.hypot((x[triangles.T[i]] - x[triangles.T[(i+1)%3]]), (y[triangles.T[i]] - y[triangles.T[(i+1)%3]]))
            max_edge = np.max(d.T, axis=1)
            out = [edge > radio for edge in max_edge]
            return out

        if symmetry: 
            x_mat -= np.median(x_mat, axis=1).reshape(n_steps, 1)
            if type(radius) != NoneType:
                if type(radius) == FunctionType:
                    radius = radius(tarr)
                tt, x_mat = warp_circle(tarr, x_mat, radius, np.ptp(x_mat[0]), tarr[-1]/ratio, flipx) 

        tt_flat, x_mat_flat = tt.flatten(), x_mat.flatten()

        if flipy:
            triang = tri.Triangulation(x_mat_flat, tt_flat)
            mask = long_edges(x_mat_flat, tt_flat, triang.triangles, 1.2*np.hypot(np.max(np.diff(tarr)), np.max(np.diff(x_mat_flat))))
        else:
            triang = tri.Triangulation(tt_flat, x_mat_flat)
            mask = long_edges(tt_flat, x_mat_flat, triang.triangles, 1.2*np.hypot(np.max(np.diff(tarr)), np.max(np.diff(x_mat_flat))))
        
        triang.set_mask(mask)

        for idx in range(self.Nsys):
            uu = u_mat[:,idx,:]

            integ = np.trapz(np.trapz(uu, x_mat, axis=1), np.linspace(0, self.width / init_width * 6.35, len(tarr)))
            print("Integral", idx, ":", integ)

            ax = fig.add_subplot(1, self.Nsys, idx + 1, projection=None)
            surf = ax.tricontourf(triang, uu.flatten(), 256, cmap=self.cmaps[idx], zorder=0) 
            ax.set_aspect(scaling, adjustable='box')
            if flipy:
                ax.set_xlim(np.min(x_mat_flat), np.max(x_mat_flat))
                ax.set_ylim(0, np.max(tt))
            else:
                ax.set_xlim(0, np.max(tt))
                ax.set_ylim(np.min(x_mat_flat), np.max(x_mat_flat))
            if flipx:
                ax.invert_yaxis()
            self.make_labels(ax, surf, idx, flipy)
            # ax.set_axis_off()

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches=0, transparent=False)