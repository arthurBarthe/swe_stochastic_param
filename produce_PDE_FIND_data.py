"""
A script to run a 30km resolution model from
a ten year spin up and dump data at very high 
temporal frequency.
"""

from shallowwater import ShallowWaterModel

model = ShallowWaterModel( Nx=128, Ny=128, Lx=3840e3, Ly=3840e3, Nt=385*1000,
                 dump_freq=385.0, dump_output=True, tau0=0.2,
                 model_name='EP_', run_name='1000_time_steps' )

filename = '_eddy_permitting__10yr_spinup.nc'

u, v, eta = model.set_initial_cond( init='state',
                                    u_file= './data/u' + filename,
                                    v_file= './data/v' + filename,
                                    eta_file= './data/eta' + filename )


for i in range( model.N_iter ) :

    percent = 100.0 * float(i) / model.N_iter

    if  ( percent ) % 1 == 0 :
        print( "{}%".format( percent ) )

    u_new, v_new, eta_new = model.integrate_forward( u, v, eta )
    
    if u_new is None :
		print( "Integration finished!" )
		break
    
    u = u_new
    v = v_new
    eta = eta_new

