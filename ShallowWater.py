"""

A class implementation of the shallow water model
solved for in:

https://github.com/milankl/swm

Tom Bolton
07/11/2018

Important departures from the original model:

- Only free-slip and RK4 are implemented to reduce amount of code.
- All global variables are removed.
- The data type is now float64 for all variables for simplicity.

"""

import numpy as np                  # version 1.11.3-py36
from scipy import sparse            # version 0.19-py36
import time as tictoc
from netCDF4 import Dataset         # version 1.2.4-py36, hdf5 version 1.8.17-py36, hdf4 version 4.2.12-py36
import glob
import zipfile



class ShallowWaterModel :

    """
    The central class for an instance of the Shallow Water model.

    This class contains the following major sections of functions:

    - Initialisation (setting parameters and grid).
    - Operators (mainly for taking derivatives).
    - Integration (running the model).
    - Output (functions for saving model ouput).

    """

    def __init__(self, Nx=256, Ny=256, Lx=3840e3, Ly=3840e3, Nt=365*24*60*60, dump_freq=24*60*60, output=0 ) :

        """
        Initialise parameters for the model.
        """

        # spatial grid parameters
        self.Nx = Nx                   # number of grid points in x-direction
        self.Ny = Ny                   # number of grid points in y-direction
        self.Lx = Lx                   # x-length of domain (m)
        self.Ly = Ly                   # y-length of domain (m)
        self.dx = Lx/Nx                # horizontal grid-spacing in x-direction (m)
        self.dy = Ly/Ny                # horizontal grid-spacing in y-direction (m)

        # temporal parameters
        self.Nt = Nt                   # integration time (s)
        self.time_scheme = 'RK4'       # numerical scheme for time-stepping
        self.dump_freq = dump_freq     # frequency to dump model output (s)

        # misc parameters
        self.bc = 0                    # boundary conditions (0 = free slip)
        self.c_D = 1e-5                # bottom friction coefficient
        self.output = output           # 1 for data storage, 0 for no storage
        self.output_path = '/my/path'  # path to location to dump output

        # initialise various components of the model


    def init_grid(self) :
        """
        Initialise the grid to numerically solve shallow water equations on.

        The model is based on an Arakawa C-grid, with 4 staggered grids:
            T-grid: for eta, sits in the middle of a grid cell.
            u-grid: for u-velocities, sits in the middle of east&west edges
            v-grid: for v-velocities, sits in the middle of north&south edges
            q-grid: for vorticity, sits on corners of grid cells.
        """
        self.max_dxdy = max( self.dx, self.dy )
        self.min_NxNy = min( self.Nx, self.Ny )
        self.lat0 = 35                                              # central latitude of beta plane
        self.dA = self.dx * self.dy                                 # area of a single grid cell (m^2)
        self.dLat = 111194.0                                        # 1 degree latitude (m)
        self.dLon = self.dLat * np.cos( np.pi * self.lat0 / 180 )   # 1 degree longitude at lat0

        self.NT = self.Nx * self.Ny                                 # number of T grid points (for h)
        self.Nu = (self.Nx-1) * self.Ny                             # number of u grid points
        self.Nv = self.Nx * (self.Ny-1)                             # number of v grid points
        self.Nq = (self.Nx+1) * (self.Ny+1)                         # number of q grid points

        # grid vectors





