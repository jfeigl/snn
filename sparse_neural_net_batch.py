import numpy as np
import time

from sklearn.utils import shuffle

from utils import yield_batches
from utils import RMSE
from utils import sparsity


class Network():
    def __init__(self,
                 layers,
                 epochs,
                 batchsize):

        self.layers = layers
        self.epochs = epochs
        self.batchsize = batchsize

        self.validation = []

        self._init_layers()

    def _init_layers(self):

        # init weights
        self.layers[0].FrontLayer = None
        self.layers[0].BackLayer = self.layers[1]

        self.layers[1].FrontLayer = self.layers[0]
        self.layers[1].BackLayer = self.layers[2]

        self.layers[2].FrontLayer = self.layers[1]
        self.layers[2].BackLayer = None

        # init weights
        for l in self.layers:
            l._init_weights()

    def _update_layer_idx(self, input_idx, target_idx):

        self.layers[0].input_idx = input_idx
        self.layers[-1].target_idx = target_idx

    def predict(self, input_idx, target_idx):

        self._update_layer_idx(input_idx, target_idx)

        self._predict()
        return self.layers[-1].a

    def _predict(self):

        for l in self.layers:
            l.get_activation()

    def _evaluate(self, X_test, y_test, epoch, duration, Scaler):

        self.layers[1].b_test = True

        result = []
        target = []

        for X, y in yield_batches(X_test, y_test, self.batchsize):

            users = X[:, 0]
            movies = X[:, 1]
            ratings = y

            pred = Scaler.inv(self.predict(users, movies))

            result.extend(pred)
            target.extend(ratings)

        print(np.min(result), np.mean(result), np.max(result))
        val_error = RMSE(target, np.clip(result, 1, 5))
        self.validation.append(val_error)

        print('epoch: {:>2d} | test error: {:>1.5f} | time: {:>5.2f}'.format(
            epoch, val_error, duration))

    def fit(self, X_train, y_train):

        for i in range(self.epochs):

            # update epochs for modified rms updater
            self.layers[1].updater.iteration = i
            self.layers[2].updater.iteration = i

            X_train, y_train = shuffle(X_train, y_train)
            for X, y in yield_batches(X_train, y_train, self.batchsize):
                self._fit(X, y)

    def fit_test(self, X_train, y_train, X_test, y_test, Scaler):

        # evaluate before starting training
        self._evaluate(X_test, y_test, 0, 0, Scaler)

        for i in range(self.epochs):
            start = time.time()

            # update epochs for modified rms updater
            for l in self.layers:
                l.epoch = i

            X_train, y_train = shuffle(X_train, y_train)
            for X, y in yield_batches(X_train, y_train, self.batchsize):
                self._fit(X, y)

            duration = time.time() - start

            self._evaluate(X_test, y_test, i + 1, duration, Scaler)

    def _fit(self, X_train, y_train):

        self.layers[1].b_test = False

        users = X_train[:, 0]
        movies = X_train[:, 1]
        ratings = y_train

        self._update_layer_idx(users, movies)

        # feed forward
        self._predict()

        ##########
        # deltas
        ##########

        # output delta
        self.layers[2].get_delta(ratings)

        # hidden delta
        self.layers[1].get_delta()

        # input delta
        self.layers[0].get_delta()

        ################
        # weight changes
        ################

        for l in self.layers:
            l.update()
