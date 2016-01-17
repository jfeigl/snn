import numpy as np


class ElasticNetRegularizer():
    def __init__(self,
                 reg_l2,
                 reg_l1,
                 c):

        self.reg_l2 = reg_l2
        self.reg_l1 = reg_l1
        self.c = c

    def regularize(self, w, epoch):
        alpha = self.c # ** epoch
        l1_reg = (1. - alpha) * np.sign(w)
        l2_reg = alpha * w
        return self.reg_l1 * l1_reg + self.reg_l2 * l2_reg


class L2Regularizer():
    def __init__(self,
                 reg):

        self.reg = reg

    def regularize(self, w, **kwargs):
        return self.reg * w


class L1Regularizer():
    def __init__(self,
                 reg):

        self.reg = reg

    def regularize(self, w, **kwargs):
        return self.reg * np.sign(w)
