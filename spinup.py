"""
A script to spinup a high-resolution instance of the
shallow water model as a test.
"""

from shallowwater import ShallowWaterModel
from netCDF4 import Dataset
from utils import coarsen
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import matplotlib

output_path = '/media/arthur/DATA/NYU/simulations/'
# output_path = '/scratch/ag7531/shallowWaterModel/'
from_spinup = False
n_years = 10

domain_size = 3840
factor = 1
i_run = 'non_parameterized_'


model = ShallowWaterModel(output_path=output_path,
                          Nx=domain_size // 10 // factor,
                          Ny=domain_size // 10 // factor,
                          Lx=domain_size * 1e3,
                          Ly=domain_size * 1e3,
                          Nt=n_years*360*24*60*60,
                          dump_freq=1*24*60*60, dump_output=True, tau0=0.12,
                          model_name='eddy_permitting',
                          run_name=str(i_run))

u, v, eta = model.set_initial_cond(init='rest')

if from_spinup:
    # load high-rez simulation, coarse-grain
    u_dataset = Dataset('spinup1/u_eddy_permitting__10yr_spinup.nc')
    v_dataset = Dataset('spinup1/v_eddy_permitting__10yr_spinup.nc')
    eta_dataset = Dataset('spinup1/eta_eddy_permitting__10yr_spinup.nc')
    
    u = u_dataset.variables['u'][-1, ...]
    v = v_dataset.variables['v'][-1, ...]
    eta = eta_dataset.variables['eta'][-1, ...]
    
    u = coarsen(u, 4)
    v = coarsen(v, 4)
    eta = coarsen(eta, 4)
    
    u = np.squeeze(u.reshape((-1, 1)))
    v = np.squeeze(v.reshape((-1, 1)))
    eta = np.squeeze(eta.reshape((-1, 1)))
    model.u = u
    model.v = v
    model.eta = eta

last_percent = None

for i in range( model.N_iter ) :
    percent = int(1000.0 * float(i) / model.N_iter)
    if percent != last_percent:
        print( "{}%".format( percent / 10 ) )
        last_percent = percent
        plt.imshow(model.u2mat(u), vmin=-0.5, vmax=0.5, cmap='PuOr')
        plt.show(block=False)
        plt.draw()
        plt.pause(1)

    u_new, v_new, eta_new = model.integrate_forward( u, v, eta )
    
    if u_new is None :
        print( "Integration finished!" )
        break
    
    u = u_new
    v = v_new
    eta = eta_new
