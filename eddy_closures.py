"""
A collection of various machine learning eddy closures
as classes. Most are implemented in Keras.

Some closures are written in the form of an eddy stress
tensor such that momentum and vorticity is conserved.
"""


from keras import backend as K
from keras.models import Model, load_model

class ResConvNet() :
    """
    A residual convolutional neural network to predict
    the difference between the high-resolution momentum
    tendencies du/dt dv/dt, and the tendencies from
    the low-resolution model dU/dt dV/dt.
    """
    def __init__(self) :
        self.conv_pad = 'same'   # input and output sizes should be equal

