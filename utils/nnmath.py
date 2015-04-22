__author__ = 'johannesjurgovsky'

import numpy as np


def linear(x):
    return x


def dlinear(x):
    return np.ones(x.shape)


def tanh(x):
    return np.tanh(x)


def dtanh(x):
    tan = tanh(x)
    return (1. - tan ** 2.)


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def dsigmoid(x):
    sig = sigmoid(x)
    return sig * (1. - sig)


def softmax(x):
    e = np.exp(x)
    return e / np.sum(e)  # [:, np.newaxis]


def dsoftmax(x):
    """
    NOTE: When computing the gradient of a combination of softmax output and crossentropy error,
    use :func:`~slowlearning.dsce_error` instead.
    Computing :func:`~slowlearning.dce_error` and :func:`~slowlearning.dsoftmax` separately is computationally very inefficient
    :param x:
    :return: Jacobean matrix. partial derivatives of all outputs :func:`~slowlearning.softmax` with respect to all inputs x
    """
    p = softmax(x)
    Ds = -np.outer(p, p)
    di = np.diag_indices(len(x))
    Ds[di] = p - p ** 2.
    return Ds


def rec_error(p, y):
    return 0.5 * np.sum((p - y) ** 2.)


def drec_error(p, y):
    return (p - y)


def dlinrec_error(x, y):
    return x - y


def ceb_error(p, y):
    eps = 1e-10
    return - np.sum(y * np.log(p + eps) + (1. - y) * np.log(1. - p + eps))


def dceb_error(p, y):
    return - y * 1. / p + (1. - y) / (1. - p)


def dlogceb_error(x, y):
    p = sigmoid(x)
    return - y * (1. - p) + (1. - y) * p


def cem_error(p, y):
    eps = 1e-10
    return - np.sum(y * np.log(p + eps))


def dcem_error(p, y):
    return - y * 1. / p


def dsofcem_error(x, y):
    return softmax(x) - y


class T_Func_Type:
    TANH = "tanh"
    LOGISTIC = "logistic"


class T_OutFunc_Type:
    SOFTMAX = "softmax"
    LINEAR = "linear"
    LOGISTIC = "logistic"


class T_ErrFunc_Type:
    RECONSTRUCTION = "reconstruction"
    CROSS_ENTROPY_BINOMIAL = "cross-entropy-binomial"
    CROSS_ENTROPY_MULTINOMIAL = "cross-entropy-multinomial"


class Function(object):
    def __init__(self, f, df):
        self.f = f
        self.df = df


actFuncs = {T_Func_Type.TANH: Function(tanh, dtanh),
            T_Func_Type.LOGISTIC: Function(sigmoid, dsigmoid)}

outFuncs = {T_OutFunc_Type.SOFTMAX: Function(softmax, dsoftmax),
            T_OutFunc_Type.LINEAR: Function(linear, dlinear),
            T_OutFunc_Type.LOGISTIC: Function(sigmoid, dsigmoid)}

errFuncs = {T_ErrFunc_Type.RECONSTRUCTION: Function(rec_error, drec_error),
            T_ErrFunc_Type.CROSS_ENTROPY_BINOMIAL: Function(ceb_error, dceb_error),
            T_ErrFunc_Type.CROSS_ENTROPY_MULTINOMIAL: Function(cem_error, dcem_error)}


class OutputModel(object):
    def __init__(self, outFuncType, errFuncType):
        self.p = outFuncs[outFuncType]
        self.E = errFuncs[errFuncType]
        if outFuncType == T_OutFunc_Type.SOFTMAX and errFuncType == T_ErrFunc_Type.CROSS_ENTROPY_MULTINOMIAL:
            self.dEdx = dsofcem_error
        elif outFuncType == T_OutFunc_Type.LOGISTIC and errFuncType == T_ErrFunc_Type.CROSS_ENTROPY_BINOMIAL:
            self.dEdx = dlogceb_error
        elif outFuncType == T_OutFunc_Type.LINEAR and errFuncType == T_ErrFunc_Type.RECONSTRUCTION:
            self.dEdx = dlinrec_error
        else:
            # TODO: there used to be a lambda expression which computes dEdp * dpdx, but we can not pickle lambda expression.
            self.dEdx = "composite"

    def cost_out_grad(self, x, y):
        out = self.p.f(x)
        if self.dEdx == "composite":
            return (self.E.f(out, y), out, self.E.df(self.p.f(x), y) * self.p.df(x))
        else:
            return (self.E.f(out, y), out, self.dEdx(x, y))
