import keras.engine.topology
from keras import backend as K
import numpy as np

class LogWhiten(keras.engine.topology.Layer):
    """
    Layer for normalizing the inputs to the neural network.
    """

    # Note: this is the incorrect vector since it is base10
    x_mean_vector = [
                0.04378,
                0.3213,
                1.611,
                1.817,
                1.306,
                0.5766,
                0.1092,
                0.8212
    ]

    # Note: this is the incorrect vector since it is base10
    x_standard_deviation_vector = [
                0.1622,
                0.386,
                0.748,
                0.74238,
                0.71,
                0.5592,
                0.2605,
                0.7978
    ]

    def __init__(self, **kwargs):
        self.scaling_tensor = np.array(self.x_standard_deviation_vector).reshape((1,1,1,8))
        self.centering_tensor = np.array(self.x_mean_vector).reshape((1,1,1,8))
        super(LogWhiten, self).__init__(**kwargs)

    def build(self, input_shape):
        super(LogWhiten, self).build(input_shape)

    def call(self, x):
        x = K.tf.maximum(x,[8.0])
        x = K.tf.subtract(x, [7.0])
        x = K.tf.log(x)
        #x = K.tf.subtract(x, self.centering_tensor)
        #x = K.tf.divide(x, self.scaling_tensor)
        return x
