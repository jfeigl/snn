import numpy as np


class SparseInputLayer():
    def __init__(self,
                 nr_units,
                 BiasUpdater,
                 bias):

        self.nr_units = nr_units
        self.BiasUpdater = BiasUpdater
        self.bias = bias

        self.input_idx = None
        self.FrontLayer = None
        self.BackLayer = None

        self.epoch = 0

    def _init_weights(self):
        print('init weights for input layer')

        self.BiasUpdater._init_caches(self.bias, self.bias)

    def get_delta(self):
        self.delta = self.BackLayer.BackLayer.delta

    def update(self):
        grad_b = self.delta

        b_update = self.BiasUpdater.get_updates(layer=self,
                                                grad=grad_b,
                                                index=self.input_idx)
        # b_update += self.reg * b_update

        self.bias[self.input_idx] -= b_update

    def get_activation(self):

        self.z = self.input_idx
        self.a = self.bias[self.input_idx]


class SparseHiddenLayer():
    def __init__(self,
                 nr_units,
                 activation_fct,
                 Updater,
                 Regularizer,
                 weight_function):

        self.nr_units = nr_units
        self.activation_fct = activation_fct
        self.Updater = Updater
        self.Regularizer = Regularizer
        self.weight_function = weight_function

        self.name = 'Sparse Hidden Layer'
        self.FrontLayer = None
        self.BackLayer = None

        self.epoch = 0

    def _init_weights(self):
        print('init weights for sparse hidden layer')

        self.weights = self.weight_function(self.FrontLayer.nr_units,
                                            self.nr_units)

        self.bias = []
        self.Updater._init_caches(self.weights, self.weights)

    def get_activation(self):

        self.z = self.weights[self.FrontLayer.z]
        self.a = self.activation_fct.fn(self.z)

    def get_delta(self):

        w = self.BackLayer.weights[self.BackLayer.target_idx].T

        dx = self.activation_fct.derivative(self.z)
        self.delta = (w * self.BackLayer.delta) * dx
        # self.delta = np.multiply(w, self.BackLayer.delta) * dx.T

    def update(self):

        grad_w = self.delta.T

        w_update = self.Updater.get_updates(layer=self,
                                            grad=grad_w,
                                            index=self.FrontLayer.z,
                                            epoch=self.epoch)

        # l2 regularization
        # w_update += self.regularizer(reg=self.reg,
        #                              w=self.weights[self.FrontLayer.z],
        #                              alpha=self.c ** self.epoch)

        w_update += self.Regularizer.regularize(
            w=self.weights[self.FrontLayer.z],
            epoch=self.epoch)

        self.weights[self.FrontLayer.z] -= w_update

        # max norm regularization
        # w = self.weights[self.FrontLayer.z]
        # L = np.linalg.norm(w, ord=2, axis=1)
        # L[L <= 1.5] = 1.
        # self.weights[self.FrontLayer.z] = w * (1. / L[:, np.newaxis])


class SparseOutputLayer():
    def __init__(self,
                 nr_units,
                 activation_fct,
                 Updater,
                 BiasUpdater,
                 Regularizer,
                 weight_function,
                 bias,
                 global_bias):

        self.nr_units = nr_units
        self.activation_fct = activation_fct
        self.Updater = Updater
        self.BiasUpdater = BiasUpdater

        self.weight_function = weight_function
        self.Regularizer = Regularizer
        self.global_bias = global_bias
        self.bias = bias

        self.name = 'Sparse Output Layer'
        self.target_idx = None
        self.FrontLayer = None
        self.BackLayer = None

        self.epoch = 0

    def _init_weights(self):
        print('init weights for sparse output layer')

        self.weights = self.weight_function(self.FrontLayer.nr_units,
                                            self.nr_units).T

        self.Updater._init_caches(self.weights, self.weights)
        self.BiasUpdater._init_caches(self.bias, self.bias)

    def get_activation(self):

        b_item = self.bias[self.target_idx]
        b_user = self.FrontLayer.FrontLayer.a

        interaction = np.einsum('ij,ij->i',
                                self.FrontLayer.a,
                                self.weights[self.target_idx])
        self.z = interaction + b_item + b_user + self.global_bias

        self.a = self.activation_fct.fn(self.z)

    def get_delta(self, ratings):

        dx = self.activation_fct.derivative(self.z)
        self.delta = (self.a - ratings) * dx

    def update(self):

        grad_b = self.delta
        grad_w = (self.FrontLayer.a.T * self.delta).T

        w_update = self.Updater.get_updates(layer=self,
                                            grad=grad_w,
                                            index=self.target_idx,
                                            epoch=self.epoch)

        # regularize gradient
        w_update += self.Regularizer.regularize(
            w=self.weights[self.target_idx],
            epoch=self.epoch)

        self.weights[self.target_idx] -= w_update

        # max norm regularization
        # w = self.weights[self.target_idx]
        # L = np.linalg.norm(w, ord=2, axis=1)
        # L[L <= 1.5] = 1.
        # self.weights[self.target_idx] = w * (1. / L[:, np.newaxis])

        # update bias
        b_update = self.BiasUpdater.get_updates(layer=self,
                                                grad=grad_b,
                                                index=self.target_idx)
        # b_update += self.reg * b_update

        self.bias[self.target_idx] -= b_update
