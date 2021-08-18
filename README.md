Based on https://github.com/TomBolton/ShallowEddy by Tom Bolton, itself based on https://github.com/milankl/swm by Milan. Adds a stochastic deep learning parameterization of subgrid momentum forcing.

Entry points:
- spinup.py for the high resolution
- spinupParameterized for low resolution:
  - pass --every 0 to run without parameterization
  - pass --every 1 to run with parameterization, or any positive integer (sets how often the subgrid forcing is updated)

shallowwater.py contains the default model, shallowwaterParameterized contains the model with the parameterization
