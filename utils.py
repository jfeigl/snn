import scipy as sp
import numpy as np
import pandas as pd

from tabulate import tabulate
# from line_profiler import LineProfiler


def logloss(act, pred):
    """ Vectorised computation of logloss """

    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1 - epsilon, pred)

    # compute logloss function (vectorised)
    ll = sum(act * sp.log(pred) +
             sp.subtract(1, act) * sp.log(sp.subtract(1, pred)))
    ll = ll * -1.0 / len(act)
    return ll


def do_profile(follow=[]):
    def inner(func):
        def profiled_func(*args, **kwargs):
            try:
                profiler = LineProfiler()
                profiler.add_function(func)
                for f in follow:
                    profiler.add_function(f)
                profiler.enable_by_count()
                return func(*args, **kwargs)
            finally:
                profiler.print_stats()
        return profiled_func
    return inner


def yield_batches(X, y, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in range(0, len(X), n):
        yield X[i:i + n], y[i:i + n]


def RMSE(target, result):
    error = np.sqrt(np.mean((np.array(target) - np.array(result)) ** 2))
    return error


def MSE(target, result):
    error = 1 / 2. * (np.array(target) - np.array(result)) ** 2
    return error


def get_data(path, count):

    df = pd.read_csv(path,
                     sep=',')

    df[' movie_id'] = df[' movie_id'] - 1

    if count:
        print('take first %d users' % count)
        df = df[df['simple_user_id'] < count]

    print(np.shape(df))

    return np.array(df)


def sparsity(v):
    sqrt_n = np.sqrt(len(v))
    l1 = np.sum(np.abs(v))
    l2 = np.sqrt(np.sum(v ** 2))

    sp = (sqrt_n - (l1 / l2)) / (sqrt_n - 1.)
    return sp


def get_l2_l1_ratio(x):

    B = 1. / (np.sqrt(len(x)) - 1.)
    C = np.sqrt(np.sum(x ** 2))
    D = 1. / np.sum(np.abs(x))

    der = (1. / C) - (C * D / np.abs(x))

    return -B * x * D * der


def print_stats(layer):

        headers = [layer.name, 'Stats']

        # get row-wise sparsity
        nn_ratio = (np.sum(layer.weights >= 0) /
                    float(np.product(layer.weights.shape)))

        row_sp = [sparsity(row) for row in layer.weights[:10]]

        table = [["min Weight", np.min(layer.weights)],
                 ["max Weight", np.max(layer.weights)],
                 ["mean Weight", np.mean(layer.weights)],
                 ["sparsity", np.mean(row_sp)],
                 ["nn_ratio", nn_ratio]]

        print(tabulate(table, headers=headers))
        print('\n')

        return None

