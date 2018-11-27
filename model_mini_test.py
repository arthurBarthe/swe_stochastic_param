""""
A very short test of an instance of the ShallowWaterModel
class with default parameters.
"""


import matplotlib.pyplot as plt
from shallowwater import ShallowWaterModel


test_model = ShallowWaterModel( output_path="/home/tom/Projects/swm" )

u, v, eta = test_model.set_initial_cond()

u0 = u.copy()
v0 = v.copy()
eta0 = eta.copy()

for i in range(1000) :

    u_new, v_new, eta_new = test_model.integrate_forward( u, v, eta )

    u = u_new.copy()
    v = v_new.copy()
    eta = eta_new.copy()

    if i == 999 :
        plt.imshow( test_model.h2mat(eta), origin='lower')
        plt.title(str(i))
        plt.show()