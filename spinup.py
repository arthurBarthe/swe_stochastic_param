"""
A script to spinup a high-resolution instance of the
shallow water model as a test.
"""

from shallowwater import ShallowWaterModel

model = ShallowWaterModel( Nx=128, Ny=128, Lx=3840e3, Ly=3840e3, Nt=10*360*24*60*60,
                 dump_freq=30*24*60*60, dump_output=True, tau0=0.2,
                 model_name='eddy_permitting', run_name='_10yr_spinup' )

u, v, eta = model.set_initial_cond( init='rest' )


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
    
    

