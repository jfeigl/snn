import numpy as np


def glorot_weights(prev_units, units):

    t = 2. / (prev_units + units)
    weights = np.random.normal(loc=0.0,
                               scale=t,
                               size=(prev_units, units))

    return weights.astype(np.float32)


def small_randoms(prev_units, units):
    t = 0.01
    weights = np.random.uniform(low=-t,
                                high=t,
                                size=(prev_units, units))

    return weights.astype(np.float32)


def orthogonal_weights(prev_units, units):

    a = np.random.normal(loc=0.0,
                         scale=1.,
                         size=(prev_units, units))

    u, _, v = np.linalg.svd(a, full_matrices=False)
    # pick the one with the correct shape
    q = u if u.shape == a.shape else v
    weights = q.reshape(a.shape)

    return weights.astype(np.float32) * np.sqrt(2)

