import numpy as np

EPS = 1e-8


class SparseMomentumUpdater():
    def __init__(self,
                 eta,
                 mu):

        self.eta = eta
        self.mu = mu

    def _init_caches(self,
                     prev_update,
                     cache):

        self.prev_update = np.zeros_like(prev_update)

    def get_updates(self, layer, grad, index, **kwargs):

        prev_update = self.prev_update[index]
        update = self.mu * prev_update + self.eta * grad

        # keep weight_updates
        self.prev_update[index] = update

        return update


class SparseNesterovUpdater():
    def __init__(self,
                 eta,
                 mu):

        self.eta = eta
        self.mu = mu

    def _init_caches(self,
                     prev_update,
                     cache):

        self.prev_update = np.zeros_like(prev_update)

    def get_updates(self, layer, grad, index, **kwargs):

        prev_update = self.prev_update[index]

        self.prev_update[index] = (self.mu * self.prev_update[index]
                                   - self.eta * grad)

        update = (self.mu * prev_update -
                  (1 + self.mu) * self.prev_update[index])

        return update


class SparseAdagradUpdater():
    def __init__(self,
                 eta,
                 mu):

        self.eta = eta
        self.mu = mu

    def _init_caches(self,
                     prev_update,
                     cache):

        self.prev_update = np.zeros_like(prev_update)
        self.cache = np.zeros_like(cache)

    def get_updates(self, layer, grad, index, **kwargs):

        self.cache[index] += grad ** 2
        prev_update = self.prev_update[index]
        new_update = grad / np.sqrt(EPS + self.cache[index])

        update = self.mu * prev_update + self.eta * new_update

        # keep w_updates
        self.prev_update[index] = update

        return update


class SparseRMSPropUpdater():
    def __init__(self,
                 eta,
                 mu,
                 decay):

        self.eta = eta
        self.mu = mu
        self.decay = decay

    def _init_caches(self,
                     prev_update,
                     cache):

        self.prev_update = np.zeros_like(prev_update)
        self.cache = np.zeros_like(cache)

    def get_updates(self, layer, grad, index, **kwargs):

        self.cache[index] = (self.decay * self.cache[index] +
                             (1.0 - self.decay) * (grad ** 2))

        update = self.eta * grad / np.sqrt(EPS + self.cache[index])

        return update


class SparseRMSPropUpdaterWithMomentum():
    def __init__(self,
                 eta,
                 mu,
                 decay,
                 switch):

        self.eta = eta
        self.mu = mu
        self.decay = decay
        self.switch = switch

    def _init_caches(self,
                     prev_update,
                     cache):

        self.prev_update = np.zeros_like(prev_update)
        self.cache = np.zeros_like(cache)

    def get_updates(self, layer, grad, index, epoch):

        if epoch <= self.switch:

            self.cache[index] = (self.decay * self.cache[index] +
                                 (1.0 - self.decay) * (grad ** 2))

            update = self.eta * grad / np.sqrt(EPS + self.cache[index])

        else:
            prev_update = self.prev_update[index]
            update = self.mu * prev_update + self.eta * grad

            # keep weight_updates
            self.prev_update[index] = update

        return update


class SparseMomentumDropoutUpdater():
    def __init__(self,
                 eta,
                 mu):

        self.eta = eta
        self.mu = mu

    def _init_caches(self,
                     prev_update,
                     cache):

        self.prev_update = np.zeros_like(prev_update)

    def get_updates(self, layer, grad, index, **kwargs):

        # import pdb; pdb.set_trace()
        prev_update = self.prev_update[index]
        update = self.mu * prev_update + self.eta * grad

        # keep weight_updates
        new = self.prev_update[index]
        new[layer.dropout_mask] = update[layer.dropout_mask]

        self.prev_update[index] = new

        return update * layer.dropout_mask


class SparseClippedRMSPropUpdater():
    def __init__(self,
                 eta,
                 mu,
                 decay):

        self.eta = eta
        self.mu = mu
        self.decay = decay

    def _init_caches(self,
                     prev_update,
                     cache):

        self.prev_update = np.zeros_like(prev_update)
        self.cache = np.zeros_like(cache)

    def get_updates(self, layer, grad, index, **kwargs):

        self.cache[index] = (self.decay * self.cache[index] +
                             (1.0 - self.decay) * (grad ** 2))

        l_rate = self.eta / np.sqrt(EPS + self.cache[index])
        l_rate = np.clip(l_rate, 0.001, 0.05)
        update = l_rate * grad

        return update
