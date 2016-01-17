import numpy as np
from scipy.special import expit
from scipy.misc import logsumexp


class Sigmoid():

    @staticmethod
    def fn(x):
        return expit(x)

    @staticmethod
    def derivative(x):
        x = Sigmoid.fn(x)
        return x * (1. - x)


class Tanh():

    @staticmethod
    def fn(x):
        return np.tanh(x)

    @staticmethod
    def derivative(x):
        x = Tanh.fn(x)
        return 1. - (x ** 2.)


class Linear():

    @staticmethod
    def fn(x):
        return x

    @staticmethod
    def derivative(x):
        return 1.


class ClippedLinear():

    @staticmethod
    def fn(x):
        return np.clip(x, -1., 1.)

    @staticmethod
    def derivative(x):
        return np.where(abs(x) <= 1, 1, 0)


class ReLu():

    @staticmethod
    def fn(x):
        return x.clip(0)

    @staticmethod
    def derivative(x):
        return x > 0


class ScaledTanh():

    @staticmethod
    def fn(x):
        return 1.7159 * np.tanh((2. / 3.) * x)

    @staticmethod
    def derivative(x):
        x = ScaledTanh.fn(x)
        return (0.6667 / 1.7159) * (1.7159 - x) * (1.7159 + x)

