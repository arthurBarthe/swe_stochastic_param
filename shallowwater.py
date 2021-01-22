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

import numpy as np
import pickle
from scipy import sparse
from netCDF4 import Dataset


class ShallowWaterModel :

    """
    The central class for an instance of the Shallow Water model.

    This class contains the following major sections of functions:

    - Initialisation (setting parameters and grid).
    - Operators (for taking derivatives and interpolating).
    - Integration (running the model).

    """

    ####################################################################################################################
    #
    # INITIALISATION
    #
    ####################################################################################################################

    def __init__(self, output_path='', Nx=256, Ny=256, Lx=3840e3, Ly=3840e3, Nt=360*24*60*60, 
                 dump_freq=1*24*60*60, dump_output=False, tau0=0.12,
                 nu_lap=540, model_name='',
                 run_name='0001', bc=2) :

        """
        Initialise parameters for the model.
        """
        print( "--------------------------------------------" )
        print( "Initialising various components of the model" )
        print( "--------------------------------------------" )

        # spatial grid parameters
        self.Nx = Nx                    # number of grid points in x-direction
        self.Ny = Ny                    # number of grid points in y-direction
        self.Lx = Lx                    # x-length of domain (m)
        self.Ly = Ly                    # y-length of domain (m)
        self.dx = Lx/float(Nx)          # horizontal grid-spacing in x-direction (m)
        self.dy = Ly/float(Ny)          # horizontal grid-spacing in y-direction (m)

        # temporal parameters
        self.Nt = Nt                    # integration time (s)
        self.time_scheme = 'RK4'        # numerical scheme for time-stepping
        self.dump_freq = dump_freq      # frequency to dump model output (s)
        self.dump_iter = 0

        # misc parameters
        self.bc = bc                     # boundary conditions (2 = no slip)
        self.c_D = 1e-5                    # bottom friction coefficient -- was 1e-5
        self.nu_lap = nu_lap            # laplacian viscosity coefficient
        self.run_name = run_name        # name of this particular integration
        self.model_name = model_name    # name of this model        
        self.dump_output = dump_output  # bool for whether to dump output
        self.output_path = output_path  # where to store model output
        self.tau0 = tau0                # wind stress forcing amplitude

        # initialise various components of the model
        self.init_grid();              print("--> Grid initialised.")
        self.set_coriolis();           print("--> Coriolis calculated.")
        self.set_viscosity();          print("--> Viscosity initialised.")
        self.set_forcing();            print("--> Wind forcing calculated")
        self.set_timestep();           print("--> Time-step calculated.")

        # initialise all operators of the model
        self.init_grad_matrices();     print("--> Gradient matrices initialised.")
        self.set_lapl_matrices();      print("--> Laplacian matrices initialised.")
        self.set_interp_matrices();    print("--> Interpolation matrices initialised.")
        self.set_arakawa_matrices();   print("--> Arakawa matrices initialised.")

        # only configure output if needed
        if dump_output : self.config_output();    print("--> Configured output settings.")
        
        print( "\nDone! Ready to set initial conditions.")



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
        
        print("\t ...init_grid:: numerical discretisation of {}x{} grid points.".format( self.Nx, self.Ny ) )
        print("\t ...init_grid:: horizontal resolution of dx = {} km, dy = {} km.".format( int( self.dx*0.001), int( self.dy*0.001 ) ) )
        

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
        
        print("\t ...set_coriolis:: central latitude of beta-plane lat0 = {}N.".format( self.lat0 ) )

    def set_viscosity(self) :
        """
        Linear scaling of constant viscosity coefficients
        based on nu_lap = 540 (m^2s^-1) at 30km resolution.
        """
        self.nu_bih = self.nu_lap * self.max_dxdy ** 2
        print("\t ...set_viscosity:: laplacian coefficient {}, biharmonic coefficient {:.1E}.".format( self.nu_lap, self.nu_bih) )

    def set_forcing(self) :
        """
        Calculate the constant zonal wind forcing, which includes the density
        rho, but excludes the 1/h (which is included in the model time-stepping).
        """

        self.rho = 1e3      # density of water (kgm^-3)
        xx_u, yy_u = np.meshgrid( self.x_u, self.y_u )

        # meridional width of forcing pattern
        sigma = 100e3

        # self.tau_x = self.tau0 * np.exp( - 0.5 * np.square( ( yy_u - self.Ly * 0.5 ) / sigma ) ) / self.rho
        self.tau_x = self.tau0*(np.cos(2*np.pi*(yy_u-self.Ly/2)/self.Ly) \
                                + 2*np.sin(np.pi*(yy_u - self.Ly/2)/self.Ly)) \
            / self.rho

        self.tau_x = self.tau_x.flatten()
        
        print("\t ...set_forcing:: wind forcing amplitude of {} Nm-2.".format( self.tau0 ) )

    def set_timestep(self) :
        """
        Set time-step of model such that CFL stability is
        respected and gravity waves are resolved.
        """
        self.c_phase = np.sqrt( self.g * self.H )                               # gravity wave speed
        self.dt = np.floor( ( 0.9 * min( self.dx, self.dy ) ) / self.c_phase )  # time-step (s)
        self.N_iter = int( np.ceil( self.Nt / self.dt ) )                       # number of iterations/time-steps
        self.t = 0                                                              # current time (s)
        self.iter = 0                                                           # current iteration
        
        print("\t ...set_timestep:: calculated timestep is dt = {} seconds.".format( self.dt ) )
        print("\t ...set_timestep:: total number of iterations to run for is N_iter = {}.".format( int( self.N_iter ) ) )

    def set_initial_cond(self, init='rest', u_file=None, v_file=None, eta_file=None ) :
        """
        Initialise the prognostic variables of the model either
        
        :param init: 'rest' -> run model from scratch from rest
                     'state' -> run model from given u, v, eta

        :param u_init:     2D array of u field
        :param v_init:     2D array of v field
        :param eta_init:   2D array of eta field
        
        :return: u_0, v_0, eta_0
        """

        if init == 'rest' :

            u_0 = np.zeros( self.Nu )
            v_0 = np.zeros( self.Nv )
            eta_0 = np.zeros( self.NT )

        elif init == 'state' :

            # load from .nc file
            u_data = Dataset( u_file )
            v_data = Dataset( v_file ) 
            eta_data = Dataset( eta_file )
            
            u = np.array( u_data.variables['u'] )
            v = np.array( v_data.variables['v'] )
            eta = np.array( eta_data.variables['eta'] )
            
            if len( u.shape ) == 3 :
                np.squeeze( u[-1,:,:].reshape( (-1,1) ) )
                np.squeeze( v[-1,:,:].reshape( (-1,1) ) )
                eta_0 = np.squeeze( eta[-1,:,:].reshape( (-1,1) ) )
            else : 
                u_0 = np.squeeze( u.reshape( (-1,1) ) )
                v_0 = np.squeeze( v.reshape( (-1,1) ) )
                eta_0 = np.squeeze( eta.reshape( (-1,1) ) )
            
        # keep a copy of the most recent values 
        # of each field
        self.u = u_0.copy()
        self.v = v_0.copy()
        self.eta = eta_0.copy()

        return u_0, v_0, eta_0



    def config_output(self) :
        """
        Configure where to saved model output and initialise the nc-files.
        """
        self.N_dumps = np.floor( self.Nt / float( self.dump_freq ) )
        self.output_iter_freq = np.floor( self.dump_freq / self.dt )          # iteration frequency of data dumping
        self.true_dump_freq = np.ceil( self.dt * self.output_iter_freq )      # true dump frequency in seconds

        if self.dump_output :

            self.dump_iter = 0   

            # store files, dimensions and variables in dictionnaries
            self.ncu = dict()
            self.ncv = dict()
            self.ncdu = dict()
            self.ncdv = dict()
            self.nceta = dict()
            if hasattr(self, 's_x'):
                self.ncdu = dict()
                self.ncdv = dict()

            # creating the netcdf files
            ncformat = 'NETCDF4'
            filename = self.model_name + '_' + self.run_name + '.nc'
            self.ncu['file'] = Dataset( self.output_path + 'u_' + filename, 'w', format=ncformat )
            self.ncv['file'] = Dataset( self.output_path + 'v_' + filename, 'w', format=ncformat )
            self.nceta['file'] = Dataset( self.output_path + 'eta_' + filename, 'w', format=ncformat )

            self.ncdu['file'] = Dataset( self.output_path + 'du_' + filename, 'w', format=ncformat )
            self.ncdv['file'] = Dataset( self.output_path + 'dv_' + filename, 'w', format=ncformat )

            
            print( "\t ...config_output:: files to be written to {}.".format( self.output_path ) )
            print( "\t ...config_output:: model data be dumped every {} seconds.".format( int( self.true_dump_freq ) ) )

            params = [ 'rho', 'tau0', 'dt', 'nu_lap', 'nu_bih', 'lat0', 'f0', 'beta', 'H', 'c_D',
                      'Nx', 'Ny', 'dx', 'dy', 'N_dumps', 'dump_freq', 'true_dump_freq', 'bc']

            for p in params:
                self.ncu['file'].setncattr( p, getattr( self, p ) )
                self.ncv['file'].setncattr( p, getattr( self, p ) )
                self.nceta['file'].setncattr( p, getattr( self, p ) )
                self.ncdu['file'].setncattr( p, getattr( self, p ) )
                self.ncdv['file'].setncattr( p, getattr( self, p ) )

            # create dimensions
            self.ncu['xdim'] = self.ncu['file'].createDimension( 'x', self.Nx-1 )
            self.ncu['ydim'] = self.ncu['file'].createDimension( 'y', self.Ny )
            self.ncu['tdim'] = self.ncu['file'].createDimension( 't', None )

            self.ncv['xdim'] = self.ncv['file'].createDimension( 'x', self.Nx )
            self.ncv['ydim'] = self.ncv['file'].createDimension( 'y', self.Ny-1 )
            self.ncv['tdim'] = self.ncv['file'].createDimension( 't', None )

            self.nceta['xdim'] = self.nceta['file'].createDimension( 'x', self.Nx )
            self.nceta['ydim'] = self.nceta['file'].createDimension( 'y', self.Ny )
            self.nceta['tdim'] = self.nceta['file'].createDimension( 't', None )
            
            self.ncdu['xdim'] = self.ncdu['file'].createDimension( 'x', self.Nx-1 )
            self.ncdu['ydim'] = self.ncdu['file'].createDimension( 'y', self.Ny )
            self.ncdu['tdim'] = self.ncdu['file'].createDimension( 't', None )
            
            self.ncdv['xdim'] = self.ncdv['file'].createDimension( 'x', self.Nx )
            self.ncdv['ydim'] = self.ncdv['file'].createDimension( 'y', self.Ny-1 )
            self.ncdv['tdim'] = self.ncdv['file'].createDimension( 't', None )

            # create variables
            p = 'f8' # 32-bit precision storing, or f8 for 64bit
            for var in ['u','v','eta'] :
        
                nc_dict = getattr( self, 'nc'+var )
                
                nc_dict['t'] = nc_dict['file'].createVariable( 't', p, ('t',), zlib=True, fletcher32=True )
                nc_dict['x'] = nc_dict['file'].createVariable( 'x', p, ('x',), zlib=True, fletcher32=True )
                nc_dict['y'] = nc_dict['file'].createVariable( 'y', p, ('y',), zlib=True, fletcher32=True )
                nc_dict[var] = nc_dict['file'].createVariable( var, p, ('t','y','x'), zlib=True, fletcher32=True )
            
            for var in ['du','dv'] :
                nc_dict = getattr( self, 'nc'+var )
                
                nc_dict['t'] = nc_dict['file'].createVariable( 't', p, ('t',), zlib=True, fletcher32=True )
                nc_dict['x'] = nc_dict['file'].createVariable( 'x', p, ('x',), zlib=True, fletcher32=True )
                nc_dict['y'] = nc_dict['file'].createVariable( 'y', p, ('y',), zlib=True, fletcher32=True )
                nc_dict[var] = nc_dict['file'].createVariable( var, p, ('t','y','x'), zlib=True, fletcher32=True )
                if var == 'du':
                    nc_dict['s_x'] = nc_dict['file'].createVariable('s_x', p, ('t','y','x'), zlib=True, fletcher32=True )
                elif var == 'dv':
                    nc_dict['s_y'] = nc_dict['file'].createVariable('s_y', p, ('t','y','x'), zlib=True, fletcher32=True )

            self.ncu['u'].units = 'm/s'
            self.ncv['v'].units = 'm/s'
            self.nceta['eta'].units = 'm'
            self.ncdu['du'].units = 'm/s^2'
            self.ncdv['dv'].units = 'm/s^2'

            # write dimensions
            for var1, var2 in zip( ['u', 'v', 'eta' ], [ 'u', 'v', 'T' ] ) :
                nc_dict = getattr( self, 'nc'+var1 )
                nc_dict['x'][:] = getattr( self, 'x_'+var2 )
                nc_dict['y'][:] = getattr( self, 'y_'+var2 )

            for var1, var2 in zip( ['du', 'dv'], [ 'u', 'v'] ) :
                nc_dict = getattr( self, 'nc'+var1 )
                nc_dict['x'][:] = getattr( self, 'x_'+var2 )
                nc_dict['y'][:] = getattr( self, 'y_'+var2 )

    def update_nc_files(self) :
        """
        Dump data at current timestep into .nc files.
        """
        if ( self.iter + 1 ) % self.output_iter_freq == 0:
            
            I = self.dump_iter 

            self.ncu['t'][I] = self.t
            self.ncu['u'][I,:,:] = self.u2mat( self.u )

            self.ncv['t'][I] = self.t
            self.ncv['v'][I,:,:] = self.v2mat( self.v )

            self.nceta['t'][I] = self.t
            self.nceta['eta'][I,:,:] = self.h2mat( self.eta )

            if hasattr(self, 's_x'):
                self.ncdu['t'][I] = self.t
                self.ncdu['s_x'][I, :, :] = self.u2mat(self.s_x)
                self.ncdu['du'][I, :, :] = self.u2mat(self.du)
                self.ncdv['t'][I] = self.t
                self.ncdv['s_y'][I, :, :] = self.v2mat(self.s_y)
                self.ncdv['dv'][I, :, :] = self.v2mat(self.dv)

            self.dump_iter += 1
    
            print( "\t ...integrate_forward:: dumped output for {}th time at iteration {}.".format( int( self.dump_iter ),  int( self.iter ) ) )
                
            # check if finished dumping output
            if (self.dump_iter + 1 ) == self.N_dumps :
                self.ncu['file'].close()
                self.ncv['file'].close()
                self.nceta['file'].close()
                self.ncdu['file'].close()
                self.ncdv['file'].close()
                print( "\t ...integrate_forward:: All output has been dumped into the .nc files.")
                if hasattr(self, 's_x'):
                    print('\t... also saving the forcing')

    ####################################################################################################################
    #
    # OPERATORS
    #
    ####################################################################################################################

    def h2mat(self,eta) :
        return eta.reshape( ( self.Ny, self.Nx ) )

    def u2mat(self,u) :
        return u.reshape( ( self.Ny, self.Nx-1 ) )

    def v2mat(self,v) :
        return v.reshape( ( self.Ny-1, self.Nx ) )

    def q2mat(self,q) :
        return q.reshape( ( self.Ny+1, self.Nx+1 ) )

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
        self.Gvy = - self.GTy.T.tocsr()      # d/dy from v to T grid

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
        Construct all 2- or 4-point interpolation matrices
        between the u, v, T and q grids.

        I   -> shorthand for interpolation matrix

        E.g. Iuv is the interpolation matrix from u- to v-grid

        """

        # index used to delete the rows that correspond to
        # an interpolation across the east-west boundaries,
        # i.e. to remove the periodicity in x
        indx1 = list( range( self.Nv + self.Nx ) );      del indx1[ ( self.Nx - 1 )::self.Nx ]
        indx2 = list( range( self.NT + self.Ny - 1 ) );  del indx2[ self.Nx::( self.Nx + 1 ) ]

        # interpolate v-points to u-points, 4-point average
        # including the information of the kinematic boundary condition
        d = np.ones( self.Nv ) / 4.  # diagonal
        self.Ivu = ( sparse.dia_matrix( (d, 0), shape=( self.Nv + self.Nx, self.Nv ) ) +
                     sparse.dia_matrix( (d, 1), shape=( self.Nv + self.Nx, self.Nv ) ) +
                     sparse.dia_matrix( (d, -self.Nx + 1), shape=( self.Nv + self.Nx, self.Nv ) ) +
                     sparse.dia_matrix( (d, -self.Nx), shape=( self.Nv + self.Nx, self.Nv ) ) )[indx1, :]

        # interpolate u-points to v-points, 4-point average
        # including the information of the kinematic boundary condition
        self.Iuv = self.Ivu.T

        # interpolate q-points to T-points, 4-point average
        d = np.ones( self.Nq + self.Ny - 1 ) / 4.  # diagonal
        self.IqT = ( sparse.dia_matrix( (d, 0), shape=( ( self.NT + self.Ny - 1, self.Nq ) ) ) +
                     sparse.dia_matrix( (d, 1), shape=( ( self.NT + self.Ny - 1, self.Nq ) ) ) +
                     sparse.dia_matrix( (d, self.Nx + 1), shape=( ( self.NT + self.Ny - 1, self.Nq ) ) ) +
                     sparse.dia_matrix( (d, self.Nx + 2), shape=( ( self.NT + self.Ny - 1, self.Nq ) ) ) )[indx2, :]

        self.IuT = abs( self.Gux * self.dx / 2.)   # interpolate u-points to T-points, 2-point average
        self.IvT = abs( self.Gvy * self.dy / 2.)   # interpolate v-points to T-points, 2-point average
        self.ITu = abs( self.GTx * self.dx / 2.)   # interpolate T-points to u-points, 2-point average
        self.ITv = abs( self.GTy * self.dy / 2.)   # interpolate T-points to v-points, 2-point average

        # interpolate q-points to u-points, 2-point average
        d = np.ones( self.Nq ) / 2.
        indx3 = list( range( self.Nq - self.Nx - 1) ); del indx3[::(self.Nx + 1)]; del indx3[self.Nx - 1::self.Nx]
        self.Iqu = ( sparse.dia_matrix( (d, 0), shape=( ( self.Nq - self.Nx - 1, self.Nq ) ) ) +
                     sparse.dia_matrix(( d, self.Nx + 1), shape=( ( self.Nq - self.Nx - 1, self.Nq ) ) ) )[indx3, :]

        # interpolate u-points to q-points, 2-point average
        # include lateral boundary condition information in Iqu.T
        self.Iuq = self.Iqu.T.tocsr().copy()
        self.Iuq.data[:self.Nx - 1] = 1 - self.bc / 2.
        self.Iuq.data[-self.Nx + 1:] = 1 - self.bc / 2.

        # interpolate q-points to v-points, 2-point average
        # same diagonal d as for Iqu, reuse
        indx4 = list( range( self.Nv + self.Ny - 2) ); del indx4[self.Nx::(self.Nx + 1)]
        self.Iqv = ( sparse.dia_matrix( (d, self.Nx + 1), shape=( ( self.Nv + self.Ny - 2, self.Nq ) ) ) +
                     sparse.dia_matrix( (d, self.Nx + 2), shape=( ( self.Nv + self.Ny - 2, self.Nq ) ) ) )[indx4, :]

        # interpolate v-points to q-points, 2-point average
        # include lateral boundary condition information in Iqv.T
        self.Ivq = self.Iqv.T.tocsr().copy()
        self.Ivq.data[::2 * self.Nx] = 1 - self.bc / 2.
        self.Ivq.data[::-2 * self.Nx] = 1 - self.bc / 2.

        # interpolate T-points to q-points, copy T points to ghost points (no h gradients across boundaries)
        # data vector with entries increased by *2,*4 for ghost-point copy
        # equivalently: ITq.sum(axis=1) must be 1 in each row.
        d = np.ones(4 * self.NT ) / 4.
        d[1:2 * self.Nx - 1] = .5
        d[-2 * self.Nx + 1:-1] = .5
        d[2 * self.Nx:-2 * self.Nx - 1:4 * self.Nx] = .5
        d[2 * self.Nx + 1:-2 * self.Nx - 1:4 * self.Nx] = .5
        d[-2 * self.Nx - 1:2 * self.Nx:-4 * self.Nx] = .5
        d[-2 * self.Nx - 2:2 * self.Nx:-4 * self.Nx] = .5
        d[[0, 2 * self.Nx - 1, -(2 * self.Nx), -1]] = 1

        self.ITq = sparse.csr_matrix( self.IqT.T )
        self.ITq.data = d

    def set_arakawa_matrices(self) :
        """
        Set up the linear combinations of potential vorticity as in
        Arakawa and Lamb 1981.
        """

        d = np.ones( self.Nq + self.Ny - 1 ) / 24.  # data vector
        indx1 = list( range( self.Nq - self.Nx - 1 ) );  del indx1[self.Nx::(self.Nx + 1)]
        self.AL1 = ( sparse.dia_matrix( (2 * d, 0), shape=( self.Nq + self.Nx, self.Nq ) ) +
                     sparse.dia_matrix( (d, 1), shape=( self.Nq + self.Nx, self.Nq ) ) +
                     sparse.dia_matrix( (d, self.Nx + 1), shape=( self.Nq + self.Nx, self.Nq ) ) +
                     sparse.dia_matrix( (2 * d, self.Nx + 2), shape=( self.Nq + self.Nx, self.Nq ) ) )[indx1, :]

        self.indx_au = slice( -self.Nx )
        self.indx_du = slice( self.Nx, None )
        self.indx_av = list( range( self.Nx ** 2 ) );           del self.indx_av[self.Nx - 1::self.Nx]
        self.indx_dv = list( range( 1, self.Nx ** 2 + 1 ) );    del self.indx_dv[ self.Nx - 1::self.Nx]
        self.indx_av = np.array( self.indx_av )  # convert to numpy array for speed up
        self.indx_dv = np.array( self.indx_dv )

        self.AL2 = ( sparse.dia_matrix( (d, 0), shape=( self.Nq + self.Nx, self.Nq ) ) +
                     sparse.dia_matrix( (2 * d, 1), shape=( self.Nq + self.Nx, self.Nq ) ) +
                     sparse.dia_matrix( (2 * d, self.Nx + 1), shape=( self.Nq + self.Nx, self.Nq ) ) +
                     sparse.dia_matrix( (d, self.Nx + 2), shape=( self.Nq + self.Nx, self.Nq ) ) )[indx1, :]

        self.indx_bu = self.indx_du
        self.indx_cu = self.indx_au
        self.indx_bv = self.indx_dv
        self.indx_cv = self.indx_av

        indx2 = list( range( self.NT + self.Ny - 1 ) );   del indx2[self.Nx::(self.Nx + 1)];   del indx2[self.Nx - 1::self.Nx]
        self.ALeur = ( sparse.dia_matrix( (d, 0), shape=( self.NT + self.Ny - 1, self.Nq ) ) +
                       sparse.dia_matrix( (d, 1), shape=( self.NT + self.Ny - 1, self.Nq ) ) +
                       sparse.dia_matrix( (-d, self.Nx + 1), shape=( self.NT + self.Ny - 1, self.Nq ) ) +
                       sparse.dia_matrix( (-d, self.Nx + 2), shape=( self.NT + self.Ny - 1, self.Nq ) ) )[indx2, :]

        # ALeul is the epsilon linear combination to the left of the associated u-point
        indx3 = list( range( self.NT + self.Ny - 1) );   del indx3[self.Nx::(self.Nx + 1)];   del indx3[::self.Nx]
        self.ALeul = ( sparse.dia_matrix( (-d, 0), shape=( self.NT + self.Ny - 1, self.Nq ) ) +
                       sparse.dia_matrix( (-d, 1), shape=( self.NT + self.Ny - 1, self.Nq ) ) +
                       sparse.dia_matrix( (d, self.Nx + 1), shape=( self.NT + self.Ny - 1, self.Nq ) ) +
                       sparse.dia_matrix( (d, self.Nx + 2), shape=( self.NT + self.Ny - 1, self.Nq ) ) )[indx3, :]

        # Seur, Seul are shift-matrices so that the correct epsilon term is taken for the associated u-point
        ones = np.ones( self.Nu )
        ones[::(self.Nx - 1)] = 0
        self.Seur = sparse.dia_matrix( (ones, 1), shape=( self.Nu, self.Nu ) ).tocsr()
        self.Seul = self.Seur.T.tocsr()

        # Shift matrices for the a,b,c,d linear combinations
        ones = np.ones( self.Nu + self.Ny)
        indx4 = list( range( self.Nv + self.Nx ) );   del indx4[(self.Nx - 1)::self.Nx]

        self.Sau = sparse.dia_matrix( (ones, 1), shape=( self.Nu + self.Ny, self.Nv ) ).tocsr()[indx4, :]
        self.Scu = sparse.dia_matrix( (ones, 0), shape=( self.Nu + self.Ny, self.Nv ) ).tocsr()[indx4, :]
        self.Sbu = sparse.dia_matrix( (ones, -( self.Nx - 1)), shape=( self.Nu + self.Ny, self.Nv ) ).tocsr()[indx4, :]
        self.Sdu = sparse.dia_matrix( (ones, -self.Nx ), shape=( self.Nu + self.Ny, self.Nv ) ).tocsr()[indx4, :]

        ## V-component of advection
        # ALpvu is the p linear combination, up
        indx5 = list( range( self.Nq - self.Nx - 1) );   del indx5[self.Nx::(self.Nx + 1)];   del indx5[-self.Nx:]
        self.ALpvu = ( sparse.dia_matrix((-d, 0), shape=( self.Nq, self.Nq ) ) +
                       sparse.dia_matrix((d, 1), shape=( self.Nq, self.Nq ) ) +
                       sparse.dia_matrix((-d, self.Nx + 1), shape=( self.Nq, self.Nq ) ) +
                       sparse.dia_matrix((d, self.Nx + 2), shape=( self.Nq, self.Nq ) ) )[indx5, :]

        # ALpvd is the p linear combination, down
        indx6 = list( range( self.Nq - self.Nx - 1) );   del indx6[self.Nx::(self.Nx + 1)];   del indx6[:self.Nx]
        self.ALpvd = ( sparse.dia_matrix( (d, 0), shape=( self.Nq, self.Nq ) ) +
                       sparse.dia_matrix( (-d, 1), shape=( self.Nq, self.Nq ) ) +
                       sparse.dia_matrix( (d, self.Nx + 1), shape=( self.Nq, self.Nq ) ) +
                       sparse.dia_matrix( (-d, self.Nx + 2), shape=( self.Nq, self.Nq ) ) )[indx6, :]

        # associated shift matrix
        ones = np.ones( self.Nv )
        self.Spvu = sparse.dia_matrix( ( ones, self.Nx ), shape=( self.Nv, self.Nv ) ).tocsr()
        self.Spvd = self.Spvu.T.tocsr()

        # Shift matrices for a,b,c,d linear combinations
        ones = np.ones( self.Nv + self.Nx )
        indx7 = list( range( self.Nv + self.Nx ) );   del indx7[(self.Nx - 1)::self.Nx]

        self.Sav = sparse.dia_matrix( ( ones, -self.Nx ), shape=( self.Nv + self.Nx, self.Nv ) ).tocsr()[indx7, :].T.tocsr()
        self.Sbv = sparse.dia_matrix( ( ones, -(self.Nx - 1) ), shape=( self.Nv + self.Nx, self.Nv ) ).tocsr()[indx7, :].T.tocsr()
        self.Scv = sparse.dia_matrix( (ones, 0), shape=( self.Nv + self.Nx, self.Nv ) ).tocsr()[indx7, :].T.tocsr()
        self.Sdv = sparse.dia_matrix( ( ones, 1 ), shape=( self.Nv + self.Nx, self.Nv) ).tocsr()[indx7, :].T.tocsr()


    ####################################################################################################################
    #
    # INTEGRATION
    #
    ####################################################################################################################

    def rhs(self,u,v,eta) :
        """
        Set of equations:

        u_t = qhv - p_x + Fx + Mx(u,v) - bottom_friction
        v_t = -qhu - p_y + My(u,v)  - bottom_friction
        eta_t = -(uh)_x - (vh)_y

        with p = .5*(u**2 + v**2) + gh, the bernoulli potential
        and q = (v_x - u_y + f)/h the potential vorticity

        using the enstrophy and energy conserving scheme (Arakawa and Lamb, 1981) and
        a biharmonic lateral mixing term based on Shchepetkin and O'Brien (1996).

        :return: du/dt, dv/dt, deta/dt
        """
        h = eta + self.H

        h_u = self.ITu.dot(h)   # h on u grid
        h_v = self.ITv.dot(h)   # h on v grid
        h_q = self.ITq.dot(h)   # h on q grid

        U, V = u*h_u, v*h_v  # volume fluxes

        KE = self.IuT.dot( u**2 ) + self.IvT.dot( v**2 ) # kinetic energy without 1/2 factor

        # bottom friction
        bfric_u = self.c_D * self.ITu.dot( np.sqrt( KE ) ) * u / h_u
        bfric_v = self.c_D * self.ITv.dot( np.sqrt( KE ) ) * v / h_v

        # potential vorticity and bernoulli potential
        q = ( self.f_q + self.Gvx.dot(v) - self.Guy.dot(u) ) / h_q
        p = 0.5 * KE + self.g * h

        # Arakawa and Lamb advection
        AL1q = self.AL1.dot(q)
        AL2q = self.AL2.dot(q)

        adv_u = self.Seur.dot( self.ALeur.dot(q) * U ) + self.Seul.dot( self.ALeul.dot(q) * U ) + \
                self.Sau.dot( AL1q[ self.indx_au ] * V ) + self.Sbu.dot( AL2q[ self.indx_bu ] * V ) + \
                self.Scu.dot( AL2q[ self.indx_cu ] * V ) + self.Sdu.dot( AL1q[ self.indx_du ] * V )

        adv_v = self.Spvu.dot( self.ALpvu.dot(q) * V ) + self.Spvd.dot( self.ALpvd.dot(q) * V ) - \
                self.Sav.dot( AL1q[ self.indx_av ] * U ) - self.Sbv.dot( AL2q[ self.indx_bv ] * U ) - \
                self.Scv.dot( AL2q[ self.indx_cv ] * U ) - self.Sdv.dot( AL1q[ self.indx_dv ] * U )

        # symmetric stress tensor S = (S11, S12, S12, -S11), store only S11, S12
        S = ( self.Gux.dot(u) - self.Gvy.dot(v), self.Gvx.dot(v) + self.Guy.dot(u))
        hS = (h * S[0], h_q * S[1])

        diff_u = ( self.GTx * hS[0] + self.Gqy * hS[1]) / h_u
        diff_v = ( self.Gqx * hS[1] - self.GTy * hS[0]) / h_v

        # biharmonic stress tensor R = (R11, R12, R12, -R11), store only R11, R12
        R = ( self.Gux.dot( diff_u ) - self.Gvy.dot( diff_v ), self.Gvx.dot( diff_v ) + self.Guy.dot( diff_u ) )
        nuhR = ( self.nu_bih * h * R[0], self.nu_bih * h_q * R[1] )

        bidiff_u = ( self.GTx.dot( nuhR[0] ) + self.Gqy.dot( nuhR[1] ) ) / h_u
        bidiff_v = ( self.Gqx.dot( nuhR[1] ) - self.GTy.dot( nuhR[0] ) ) / h_v

        ## RIGHT-HAND SIDE: ADD TERMS
        rhs_u = adv_u - self.GTx.dot(p) + self.tau_x / h_u - bidiff_u - bfric_u
        rhs_v = adv_v - self.GTy.dot(p) - bidiff_v - bfric_v
        rhs_eta = -( self.Gux.dot(U) + self.Gvy.dot(V) )

        return rhs_u, rhs_v, rhs_eta


    def integrate_forward(self,u,v,eta, with_closure=False, closure=None ) :
        """
        Numerically integrate the model forward one time-step
        using the Runga-Kutto 4th order method.
        :return: u, v, eta
        """
        # can't trigger deep copy through [:] use .copy() instead
        u_old, v_old, eta_old = u.copy(), v.copy(), eta.copy()
        u_new, v_new, eta_new = u.copy(), v.copy(), eta.copy()

        # RK4 coefficients
        rk_a = np.array( [ 1/6.0, 1/3.0, 1/3.0, 1/6.0 ] )
        rk_b = np.array( [ 0.5, 0.5, 1.0 ] )

        for rki in range(4) :

            du, dv, deta = self.rhs( u_old, v_old, eta_old )

            if with_closure :
                du, dv = closure.predict( du, dv )

            if rki < 3:  # RHS update for the next RK-step
                u_old = u + rk_b[rki] * self.dt * du
                v_old = v + rk_b[rki] * self.dt * dv
                eta_old = eta + rk_b[rki] * self.dt * deta

            # Summing all the RHS on the go
            u_new += rk_a[rki] * self.dt * du
            v_new += rk_a[rki] * self.dt * dv
            eta_new += rk_a[rki] * self.dt * deta
            
        # update most recent fields 
        # of u, v, and eta
        self.u = u_new.copy()
        self.v = v_new.copy()
        self.eta = eta_new.copy()

        # check for NaNs
        if ( np.sum( np.isnan( self.u ) ) ) != 0 : print( "integrate_forward:: NaNs present in u! :(" )
        if ( np.sum( np.isnan( self.v ) ) ) != 0 : print( "integrate_forward:: NaNs present in v! :(" )
        if ( np.sum( np.isnan( self.eta ) ) ) != 0 : print( "integrate_forward:: NaNs present in eta! :(" )
        
        # if writing to .nc files
        if self.dump_output : self.update_nc_files()
        
        # check if at the end of the integration
        if (self.dump_iter + 1 ) == self.N_dumps :
            return None, None, None
             
        # update time-step variables
        self.t += self.dt
        self.iter += 1

        return u_new, v_new, eta_new

####################################################################################################################
#
# MISC
#
####################################################################################################################


def cg( model1, model2, u1, v1, eta1 ) :
    """
    Coarse-grain the feilds u1, v1 and eta1 from
    the higher-resolution grid of model1, to the
    lower resolution grid of model2.

    NOTE: Performs square coarse-graining, so assumes Nx = Ny
          and that dx2 is some multiple of dx1.

    :param model1:          High-res model
    :param model2:          Low-res model
    :param u1:              High-res u
    :param v1:              High-res v
    :param eta1:            High-res eta
    :return: u2, v2, eta2   Coarse-grained high-res fields to low-res grid
    """

    # interpolate u and v fields to T grid
    u1 = model1.IuT.dot( u1 )
    v1 = model1.IuT.dot( v1 )

    # convert to 2D numpy arrays
    u1 = model1.u2mat( u1 )
    v1 = model1.v2mat( v1 )
    eta1 = model1.h2mat( eta1 )

    # size of coarse-graining box
    Ncg = model2.dx / model1.dx

    if ( Ncg - int(Ncg) ) != 0 :
        print( "WARNING: dx2 not integer multiple of dx1. Aborting coarse-graining.")
        return None

    # initialise arrays for coarse-grained fields
    N1 = model1.Nx
    N2 = model2.Nx

    u2 = np.zeros( (N2,N2) )
    v2 = np.zeros( (N2,N2) )
    eta2 = np.zeros( (N2,N2) )

    for i in range(N2) :
        for j in  range(N2) :

            I, J = 4*i, 4*j

            u2[j,i] = np.mean( u1[J:J+4,I:I+4], axis=(0,1) )
            v2[j,i] = np.mean( v1[J:J+4,I:I+4], axis=(0,1) )
            eta2[j,i] = np.mean( eta1[J:J+4,I:I+4],axis=(0,1) )

    # reshape back to 1D arrays (on T-grid)
    u2 = np.reshape( u2, (N2,N2) )
    v2 = np.reshape( v2, (N2,N2) )
    eta2 = np.reshape( eta2, (N2,N2) )

    # interpolate u and v back to u and v grids (from T-grid)
    u2 = model2.ITu.dot( u2 )
    v2 = model2.ITv.dot( v2 )

    return u2, v2, eta2


            

            
    
        
        








