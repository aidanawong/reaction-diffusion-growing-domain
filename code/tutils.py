"""
Utility functions from Barr's blog post
"""
from types import FunctionType, NoneType

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.axes_grid1 import make_axes_locatable

import resources.tulip_cmap
from resources.grow_func import GROW_FORWARD_INIT_WIDTH, GROW_FORWARD_CURVATURE_CONST, \
                                GROW_BACKWARD_INIT_WIDTH, GROW_BACKWARD_CURVATURE_CONST

class BaseStateSystem:
    def __init__(self):
        raise NotImplementedError()

    def initialise(self):
        raise NotImplementedError()
    
    def initialise_cmap_figure(self):
        """
        Initializes the figure of a colour map
        It will have dimensions height to length as 0.63:self.Nsys
        """
        # fig = plt.figure(figsize=plt.figaspect(0.6/self.Nsys)) # original
        fig = plt.figure(figsize=plt.figaspect(0.63/self.Nsys))
        return fig
    
    def initialise_cmap_figure_flipy(self):
        """
        Initializes the figure of a colour map
        It will have dimensions height to length as 1:0.875*self.Nsys
        """
        fig = plt.figure(figsize=plt.figaspect(1/(0.875*self.Nsys)))
        return fig

    def update(self):
        raise NotImplementedError()
    
    def run_and_retrieve(self):
        raise NotImplementedError()
    
    def make_labels(self, ax, surf, idx, flipy):
        """
        Draws the correct labels on the subplot given
        Inputs:
            ax (matplotlib.axes._axes.Axes) - axes to which labels are to be given
            surf (mpl_toolkits.mplot3d.art3d.Poly3DCollection) - plot from ax.plot_surface or ax.tricontourf
            idx (int) - index denoting which label to use in a list of labels denoted self.labels
            flipy (bool) - True if plots are to be in portrait style, False otherwise
        """
        fontsize = 16

        if flipy:
            ax.set_xlabel("Space", fontsize=fontsize)
            ax.set_ylabel("Time", fontsize=fontsize)
        else: 
            ax.set_xlabel("Time", fontsize=fontsize)
            ax.set_ylabel("Space", fontsize=fontsize)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(surf, cax=cax) # aspect=8 fraction=0.046, pad=0.04
        cbar.set_label(self.labels[idx], fontsize=fontsize)

    def plot_surf(self, filename, n_steps):
        """
        Plots the PDE in a contour plot of space vs. time in a grid-like fashion
        Inputs: 
            filename (str) - name of the image's file
            n_steps (int) - number of 
        """
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
        """
        Uses triangulation to plot the PDE in a contour plot of space vs. time
        Inputs: 
            filename (str) - name of the image's file
            n_steps (int) - number of 
            symmetry (bool) - True if the plot should be symmetric around the line when space = 0
            scaling (str) - set to auto for automatic scaling of the box size
            radius (function or None) - None if there is no curving of the plot. 
                                        function returns the radius of curvature for each line of space at a given time
            flipx (bool) - True if from left to right the time moves from maximum to minimum. False otherwise
            flipy (bool) - True if the subplots are in portrait form wih the x-axis as space and the y-axis as time, False otherwise
        """

        mask_adjustment_const = 1.2

        self.initialise()
        if flipx: 
            ratio = GROW_BACKWARD_CURVATURE_CONST
            init_width = GROW_BACKWARD_INIT_WIDTH
        else:
            ratio = GROW_FORWARD_CURVATURE_CONST
            init_width = GROW_FORWARD_INIT_WIDTH
            
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
            mask = long_edges(x_mat_flat, tt_flat, triang.triangles, mask_adjustment_const*np.hypot(np.max(np.diff(tarr)), np.max(np.diff(x_mat_flat))))
        else:
            triang = tri.Triangulation(tt_flat, x_mat_flat)
            mask = long_edges(tt_flat, x_mat_flat, triang.triangles, mask_adjustment_const*np.hypot(np.max(np.diff(tarr)), np.max(np.diff(x_mat_flat))))
        
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

            # ax.set_axis_off() # Uncomment to remove any axes or colourbars

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches=0, transparent=False)