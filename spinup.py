"""
A script to spinup a high-resolution instance of the
shallow water model as a test.
"""

from shallowwater import ShallowWaterModel, save_model
import matplotlib.pyplot as plt

model = ShallowWaterModel( Nx=128, Ny=128, Lx=3840e3, Ly=3840e3, Nt=10*300*24*60*60,
                 dump_freq=24*60*60, dump_output=True, tau0=0.2,
                 model_name='med_res', run_name='1yr_spinup' )

u, v, eta = model.set_initial_cond( init='rest' )

#plt.imshow( model.tau_x.reshape( (127,128) ) )
#plt.colorbar()
#plt.show()

for i in range( model.N_iter ) :

    percent = 100.0 * float(i) / model.N_iter

    if  ( percent ) % 1 == 0 :
        print( "{}%".format( percent ) )

    u_new, v_new, eta_new = model.integrate_forward( u, v, eta )
    
    u = u_new
    v = v_new
    eta = eta_new
    
    
save_model( model, model_name='med_res_10yr_spin.pkl', where='./ocean_models/' )

