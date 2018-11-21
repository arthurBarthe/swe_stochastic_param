"""

A class implementation of the shallow water model
solved for in:

https://github.com/milankl/swm

Tom Bolton
07/11/2018

Important departures from the original model:

- Only RK4 is implemented to reduce amount of code.
- All global variables are removed.
- The data type is float64 for all variables for simplicity.
- Model is free-slip only.

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
        self.set_timestep();           print("--> Time-step calculated.")

        # initialise all operators of the model

        # only configure output if needed
        if output : self.config_output();    print("--> Configured output settings.")



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

    def set_initial_cond(self, path_to_init_data=None, u_file=None, v_file=None, eta_file=None ) :
        """
        Initialise the prognostic variables of the model either
        from rest, or from existing nc files for u, v and eta

        :param path_to_init_data: path to directory containing init data
        :param u_file: filename of .nc file with u data
        :param v_file: filename of .nc file with v data
        :param eta_file: filename of .nc file with eta data
        :return: u_0, v_0, eta_0
        """

        if self.init == 'rest' :

            u_0 = np.zeros( self.Nu )
            v_0 = np.zeros( self.Nv )
            eta_0 = np.zeros( self.NT )
            self.t0 = 0

        elif self.init == 'file' :

            # load from .nc file
            u_0 = Dataset( path_to_init_data + u_file )
            v_0 = Dataset( path_to_init_data + v_file )
            eta_0 = Dataset( path_to_init_data + eta_file )

        return u_0, v_0, eta_0



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

            ############################################
            # !                   !                    !
            # ADD CODE HERE TO SAVE OUTPUT AS NC FILES !
            # !                   !                    !
            ############################################

    ####################################################################################################################
    #
    # OPERATORS
    #
    ####################################################################################################################

    def init_grad_matrices(self) :
        """
        The paradigm of this model is to reshape all prognostic
        variables (u,v,eta) as 1D vectors, and then gradients of
        those variables can be calculated by simply acting on the
        1D vector with a 2D matrix.

        The elements of each gradient matrix will encode the stencils
        used to estimate each derivative.

        G          ->   shorthand for gradient
        T,u,v,q    ->   grid on which gradient is being taken
        x,y        ->   axis along which gradient is calculated

        E.g. GTx is the x-derivative on the T-grid.

        Overall, we need:

        Gux, Guy
        Gvx, Gvy
        GTx, GTy
        Gqx, Gqy

        """

        # index used to delete the rows that correspond to a derivative
        # across the east-west boundaries, i.e., to remove the periodicty
        indx1 = list( range( self.NT ) )
        indx2 = list( range( self.Nv + self.Ny - 1 ) )

        del indx1[ ( self.Nx - 1 )::self.Nx ]
        del indx2[ self.Nx::( self.Nx + 1 ) ]

        ##### 1st order gradients #####

        self.GTx = ( sparse.diags( np.ones( self.NT - 1), 1 )
                   - sparse.eye( self.NT ) )[indx1,:] / self.dx        # d/dx from T to u grid

        self.GTy = ( sparse.diags( np.ones( self.Nv ), self.Nx )
                   - sparse.eye( self.NT ) )[:-self.Nx,:] / self.dy    # d/dy from T to v grid

        self.Gux = - self.GTx.T.tocsr()      # d/dx from u to T grid
        self.Guy = - self.GTy.T.tocsr()      # d/dy from v to T grid

        # d/dy from u to q grid
        d1 = np.ones( self.Nq )
        d1[::(self.Nx+1)] = 0           # du/dy = 0 at western boundary
        d1[self.Nx::(self.Nx+1)] = 0    # du/dy = 0 at eastern boundary
        indx3 = ( d1 != 0 )             # the index to remove unnecessary columns
        d1[-self.Nx:-1] = self.bc       # north and south boundary conditions

        Guy1 = sparse.diags(d1,0).tocsr()[:,indx3][:,self.Nx-1:]
        Guy2 = sparse.diags(d1[::-1],0).tocsr()[:,indx3][:,:-(self.Nx-1)]  # fliplr and flipud of Guy1

        self.Guy = ( Guy2 - Guy1 ) / self.dy

        # d/dx from v to q grid
        sj = self.Nv + self.Ny - 1      # shape of Gvx in j-direction
        d2 = np.ones(sj)                # set up the diagonal
        d2[::(self.Nx+1)] = self.bc     # east and west boundary condition

        self.Gvx = ( sparse.dia_matrix( (d2,-(self.Nx+1)), shape=( (self.Nq,sj) ) ).tocsr()[:,indx2]
                   + sparse.dia_matrix( (-d2[::-1],-(self.Nx+1) ), shape=( (self.Nq,sj))).tocsr()[:,-np.array(indx2)[::-1]-1]
                     ) / self.dx

        # d/dy from q to u grid
        d1[-self.Nx:-1] = 1
        Gqy1 = sparse.diags(d1,0).tocsr()[:,indx3][:,self.Nx-1:]
        Gqy2 = sparse.diags(d1[::-1],0).tocsr()[:,indx3][:,:-(self.Nx-1)]  # fliplr and flipud of Gqy1
        self.Gqy = ( Gqy1 - Gqy2 ).T.tocsr() / self.dy

        # d/dx from q to v grid (make use of Gvx)
        d2[::(self.Nx+1)] = 1
        self.Gqx = ( sparse.dia_matrix((d2, -(self.Nx+1)), shape=((self.Nq,sj))).tocsr()[:, indx2] +
                     sparse.dia_matrix((-d2[::-1], -(self.Nx+1)), shape=((self.Nq, sj))).tocsr()[:, -np.array(indx2)[::-1] - 1]
                    ) / self.dx
        self.Gqx = - self.Gqx.T.tocsr()

    def set_lapl_matrices(self) :
        """
        Constructs the horizontal Laplacian (harmonic diffusion)
        and also the biharmonic diffusion operator LL.
        """

        # harmonic operators
        self.Lu = self.GTx.dot( self.Gux ) + self.Gqy.dot( self.Guy )
        self.Lv = self.Gqx.dot( self.Gvx ) + self.GTy.dot( self.Gvy )
        self.LT = self.Gux.dot( self.GTx ) + self.Gvy.dot( self.GTy )
        self.Lq = self.Gvx.dot( self.Gqx ) + self.Guy.dot( self.Gqy )

        # biharmonic operators
        self.LLu = self.Lu.dot( self.Lu )
        self.LLv = self.Lv.dot( self.Lv )
        self.LLT = self.LT.dot( self.LT )
        self.LLq = self.Lq.dot( self.Lq )

    def set_interp_matrices(self) :
        """
        Construct all 2- or 4-point interpolation
        between the u, v, T and q grids.
        """









