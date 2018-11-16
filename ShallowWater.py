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

    """

    ####################################################################################################################
    #
    # INITIALISATION
    #
    ####################################################################################################################

    def __init__(self, output_path, Nx=256, Ny=256, Lx=3840e3, Ly=3840e3, Nt=365*24*60*60, dump_freq=24*60*60, output=0,
                 tau0=0.1, init='rest' ) :

        """
        Initialise parameters for the model.
        """

        print( "Initialising various components of the model.")

        # spatial grid parameters
        self.Nx = Nx                   # number of grid points in x-direction
        self.Ny = Ny                   # number of grid points in y-direction
        self.Lx = Lx                   # x-length of domain (m)
        self.Ly = Ly                   # y-length of domain (m)
        self.dx = Lx/float(Nx)         # horizontal grid-spacing in x-direction (m)
        self.dy = Ly/float(Ny)         # horizontal grid-spacing in y-direction (m)

        # temporal parameters
        self.Nt = Nt                   # integration time (s)
        self.time_scheme = 'RK4'       # numerical scheme for time-stepping
        self.dump_freq = dump_freq     # frequency to dump model output (s)

        # misc parameters
        self.bc = 0                    # boundary conditions (0 = free slip)
        self.c_D = 1e-5                # bottom friction coefficient
        self.output = output           # 1 for data storage, 0 for no storage
        self.output_path = output_path # where to store model output
        self.tau0 = tau0               # wind stress forcing amplitude
        self.init = init               # 'rest' = run from scratch, 'file' start from .nc

        # initialise various components of the model
        self.init_grid();              print("--> Grid initialised.")
        self.set_coriolis();           print("--> Coriolis calculated.")
        self.set_viscosity();          print("--> Viscosity initialised.")
        self.set_forcing();            print("--> Wind forcing calculated")
        self.config_output();          print("--> Configure output settings.")
        self.set_timestep();           print("--> Time-step calculated.")


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
        self.H = 500                                                # water depth (m)
        self.g = 9.81                                               # acceleration due to gravity (ms^-2)

        # number of points on each grid
        self.NT = self.Nx * self.Ny                                 # number of T grid points (for h)
        self.Nu = (self.Nx-1) * self.Ny                             # number of u grid points
        self.Nv = self.Nx * (self.Ny-1)                             # number of v grid points
        self.Nq = (self.Nx+1) * (self.Ny+1)                         # number of q grid points

        # grid vectors
        self.x_T = np.arange( self.dx / 2.0, self.Lx, self.dx )     # x-coords for T grid
        self.y_T = np.arange( self.dy / 2.0, self.Ly, self.dy )     # y-coords for T grid

        self.x_u = self.x_T[:-1] + self.dx / 2.0                    # x-coords for u grid
        self.y_u = self.y_T                                         # y-coords for u grid

        self.x_v = self.x_T                                         # x-coords for v grid
        self.y_v = self.y_T[:-1] + self.dy / 2.0                    # y-coords for v grid

        self.x_q = np.arange( 0, self.Lx+self.dx/2.0, self.dx )     # x-coords for q grid
        self.y_q = np.arange( 0, self.Ly+self.dy/2.0, self.dy)      # y-coords for q grid

    def set_coriolis(self) :
        """
        Set up Coriolis parameter with beta-plane
        approximation f = f0 + beta * y,
        on all (u, v, T and q) spatial grids.
        """

        # calculate the values of f0 and beta at lat0
        omega = 2 * np.pi / ( 24.0*3600.0 )      # earth's angular frequency (s^-1)
        R = 6.371e6                              # radius of rarth (m)

        self.f0 = 2 * omega * np.sin( self.lat0 * np.pi / 180.0 )
        self.beta = 2 * omega * np.cos( self.lat0 * np.pi / 180.0 ) / R

        # construct y-vectors such that lat0 is the central latitude
        Y_u = np.array( [ self.y_u - self.Ly/2.0 ] * (self.Nx-1) ).T
        Y_v = np.array( [ self.y_v - self.Ly/2.0 ] * self.Nx ).T
        Y_q = np.array( [ self.y_q - self.Ly/2.0 ] * (self.Nx+1) ).T
        Y_T = np.array( [ self.y_T - self.Ly/2.0 ] * self.Nx ).T

        # calculate coriolis parameter on all grids (only f_q truly needed)
        self.f_u = ( self.f0 + self.beta * Y_u.flatten() )
        self.f_v = ( self.f0 + self.beta * Y_v.flatten() )
        self.f_q = ( self.f0 + self.beta * Y_q.flatten() )
        self.f_T = ( self.f0 + self.beta * Y_T.flatten() )

    def set_viscosity(self) :
        """
        Linear scaling of constant viscosity coefficients
        based on nu_lap = 540 (m^2s^-1) at 30km resolution.
        """
        self.nu_lap = 300
        self.nu_bih = self.nu_lap * self.max_dxdy ** 2

    def set_forcing(self) :
        """
        Calculate the constant zonal wind forcing, which includes the density
        rho, but excludes the 1/h (which is included in the model time-stepping).
        """

        self.rho = 1e3      # density of water (kgm^-3)
        xx_u, yy_u = np.meshgrid( self.x_u, self.y_u )

        self.tau_x = ( self.tau0 * ( np.cos( 2 * np.pi * ( yy_u-self.Ly/2.0 ) / self.Ly )
                                   + 2 * np.sin( np.pi * ( yy_u-self.Ly/2.0 ) / self.Ly )
                                   ) / self.rho
                                   ).flatten()

    def set_timestep(self) :
        """
        Set time-step of model such that CFL stability is
        respected and gravity waves are resolved.
        """
        self.c_phase = np.sqrt( self.g * self.H )                               # gravity wave speed
        self.dt = np.floor( ( 0.9 * min( self.dx, self.dy ) ) / self.c_phase )  # time-step (s)
        self.N_iter = np.ceil( ( self.Nt * 3600.0 * 24.0 ) / self.dt )          # number of iterations/time-steps

    def config_output(self) :
        """
        Configure where to saved model output and initialise the nc-files.
        """
        self.N_output = np.floor( self.dump_freq / self.dt )                    # number of times output is saved
        self.true_dump_freq = np.ceil( self.N_iter / float( self.N_output ) )   # true dump frequency

        if self.output :

            # store files, dimensions and variables in dictionaries
            self.ncu   = Dataset( self.output_path+'/u.nc', 'w', format='NETCDF4' )
            self.ncv   = Dataset( self.output_path+'/v.nc', 'w', format='NETCDF4' )
            self.nceta = Dataset( self.output_path+'/eta.nc', 'w', format='NETCDF4' )

            # store a few key parameters
            #
            params = [ 'rho', 'tau0', 'dt', 'nu_lap', 'nu_bih', 'lat0', 'f0', 'beta', 'H', 'c_D',
                       'Nx', 'Ny', 'dx', 'dy' ]

            for p in params:














