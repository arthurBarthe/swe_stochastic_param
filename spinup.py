"""
A script to spinup a high-resolution instance of the
shallow water model as a test.
"""

from shallowwater import ShallowWaterModel

model = ShallowWaterModel( Nx=512, Ny=512, Lx=3840e3, Ly=3840e3, Nt=360*24*60*60, 
                 dump_freq=30*24*60*60, dump_output=True, tau0=0.3, nu_lap=100, 
                 model_name='high_res', run_name='10yr_spinup' )

u, v, eta = model.set_initial_cond( init='rest' )

for i in range( model.N_iter ) :
    
    if  int( i * model.dt ) % ( 30*24*60*60 ) == 0 :
        print( "Month {}".format( i * model.dt / 30.0*24*60*60 ) )
    
    u_new, v_new, eta_new = model.integrate_forward( u, v, eta )
    
    u = u_new
    v = v_new
    eta_new = eta_new
    
    
    

