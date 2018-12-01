"""
A script to spinup a high-resolution instance of the
shallow water model as a test.
"""

from shallowwater import ShallowWaterModel, save_model

model = ShallowWaterModel( Nx=512, Ny=512, Lx=3840e3, Ly=3840e3, Nt=360*24*60*60,
                 dump_freq=24*60*60, dump_output=True, tau0=0.3, nu_lap=100,
                 model_name='high_res', run_name='10yr_spinup' )

u, v, eta = model.set_initial_cond( init='rest' )

for i in range( model.N_iter ) :

    percent = 100.0 * float(i) / model.N_iter

    if  ( percent ) % 1 == 0 :
        print( "{}%".format( percent ) )

    
    u_new, v_new, eta_new = model.integrate_forward( u, v, eta )
    
    u = u_new
    v = v_new
    eta_new = eta_new
    
    
save_model( model, model_name='high_res_10yr_spin.pkl', where='./ocean_models/' )

