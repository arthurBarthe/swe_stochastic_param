"""


Train the machine learning eddy closure, inspired by
the multi-timestep loss function used in:

   'Prognostic Validation of a Neural Network Unified
    Physics Parameterization',

by Brenowitz and Bretherton (2018), GRL. The training
procedure here executes the following steps:


1. Coarse-grain the fields of the (previously spun-up)
   saved high-resolution model.

2. Use the coarse-grained fields as the initial conditions
   of the lower-resolution model.

3. Integrate the high-res model forward N timesteps.

4. Integrate low-res + ML model forward N timesteps.

5. At each of the N timesteps, store the tendencies
   du/dt and dv/dt of both models.

6. Calculate the residual between the high-res tendencies
   and the low-res + ML tendencies; these maps will form the
   targets of the ML algorithm.

7. Repeat 1 to 6 until the number of time-steps equals the
   batch_size for training the ML algorithm.

8. Perform stochastic gradient descent (or whatever
   optimisation algorithm is specified) to minimise loss.

9. Repeat 7 to 8 until loss is sufficiently minimised.


"""

from shallowwater import ShallowWaterModel, load_model, save_model
from eddy_closures import *


##### Training Parameters #####

N = 16                    # run models and minimise error over this many timesteps
batch_size = 128          # number of time-steps to form a batch for training
N_samples = 10000         # repeat training process for at least this many timesteps
learn_rate = 0.0001       # learning rate of stochastic gradient descent
optimiser = 'adam'        # optimiser (e.g. adam, SGD or momentum)


# load high-resolution mode
model_HR = load_model( 'my_model.pkl', './models/' )