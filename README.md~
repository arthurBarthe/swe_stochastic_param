# Shallow Water Model in Python

An adaption of the shallow water model found [in this repo](http://www.github.com/milankl/swm), with a detailed documentation found [here](http://www.github.com/milankl/swm/tree/master/docu). A thank you to Milan for writing the original model.

As in the original model, the shallow water equations are solved numerically using the Runge-Kutta 4th order method. However, there are some important differences:

* Only free-slip boundary conditions are implemented here.
* All numerical values are represented as float64.
* The model is written as a single class in `shallowwater.py`.

I made this adaption so I could access and modify multiple instances of the model simultaneously. Exploring different machine learning eddy parameterisations becomes much easier in the class-based approach. 

The code below shows how to create and run an instance of the model for a single timestep.

```python
from shallowwater import ShallowWaterModel

my_model = ShallowWaterModel( init='rest', output=0 )

u, v, eta = my_model.set_init_cond()

u_new, v_new, eta_new = my_model.integrate_forward( u, v, eta )	
```
     

