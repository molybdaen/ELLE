__author__ = 'Johannes'

import numpy as np
import theano
from numpy import random as rng
from data import MovielensDataset
from utils import utils
from scipy.io import savemat
import matplotlib.pyplot as P
import matplotlib.ticker as MT
import matplotlib.cm as CM
import cPickle

PATH_DATA_ROOT = r"../data"


def linear(x):
    return x

def dlinear(x):
    return np.ones(x.shape)

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    tan = tanh(x)
    return (1. - tan**2.)

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def dsigmoid(x):
    sig = sigmoid(x)
    return sig*(1.-sig)

def softmax(x):
    e = np.exp(x)
    return e / np.sum(e)#[:, np.newaxis]

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
    Ds[di] = p-p**2.
    return Ds

def rec_error(p, y):
    return 0.5 * np.sum((p - y)**2.)

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
    return softmax(x)-y


class T_Func_Type:
    TANH = "tanh"
    LOGISTIC = "logistic"


class T_OutFunc_Type:
    SOFTMAX = "softmax"
    LINEAR = "linear" # linear output activation function in fact corresponds to gaussian output variables
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


class Utils:
    @staticmethod
    def code1ofK(labels, K):
        KcodedLabels = []
        for l in labels:
            codedK = np.zeros(shape=(K,))
            codedK[int(l)] = 1.
            KcodedLabels.append(codedK)
        return KcodedLabels


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
            self.dEdx = "composite"

    def cost_out_grad(self, x, y):
        out = self.p.f(x)
        if self.dEdx == "composite":
            return (self.E.f(out, y), out, self.E.df(self.p.f(x), y) * self.p.df(x))
        else:
            return (self.E.f(out, y), out, self.dEdx(x, y))


class Autoencoder(object):
    def __init__(self, nvis=100, nhid=50, eta=0.1, actfunc=actFuncs[T_Func_Type.TANH]):

        self.visible_size = nvis
        self.hidden_size = nhid

        self.W = np.asarray(rng.uniform(low=-4 * np.sqrt(6. / (self.hidden_size + self.visible_size)), high=4 * np.sqrt(6. / (self.hidden_size + self.visible_size)), size=(self.hidden_size, self.visible_size)), dtype=theano.config.floatX)
        self.b1 = np.zeros(shape=(self.hidden_size,), dtype=theano.config.floatX)
        self.b2 = np.zeros(shape=(self.visible_size,), dtype=theano.config.floatX)

        self.actfunc = actfunc
        self.eta = eta

    def _encode(self, x):
        return np.dot(self.W, x) + self.b1

    def _decode(self, h):
        return np.dot(self.W.T, h) + self.b2

    def _get_rec_err(self, x, z):
        return 0.5 * np.sum((x-z)**2.)

    def _get_ce_err(self, x, z):
        eps = 1e-10
        return - np.sum((x * np.log(z + eps) + (1.-x) * np.log(1.-z + eps)))

    def init_supervised(self, nout):
        self.output_size = nout
        self.Wlabel = np.asarray(rng.uniform(low=-4 * np.sqrt(6. / (self.output_size + self.hidden_size)), high=4 * np.sqrt(6. / (self.output_size + self.hidden_size)), size=(self.output_size, self.hidden_size)), dtype=theano.config.floatX)
        self.blabel = np.zeros(shape=(self.output_size,), dtype=theano.config.floatX)
        self.OutModel = OutputModel(T_OutFunc_Type.SOFTMAX, T_ErrFunc_Type.CROSS_ENTROPY_MULTINOMIAL)

    def get_cost_grad(self, batch):

        cost = 0.
        g_W = np.zeros(self.W.shape)
        g_b1 = np.zeros(self.b1.shape)
        g_b2 = np.zeros(self.b2.shape)

        for x in batch:
            a = self._encode(x)
            h = self.actfunc.f(a)
            p = self._decode(h)

            cost += rec_error(p, x)

            deltaOut = drec_error(p, x)

            g_W += np.outer(deltaOut, h).T
            g_b2 += deltaOut

            deltaHidden = np.dot(self.W, deltaOut) * self.actfunc.df(a)
            g_W += np.outer(deltaHidden, x)
            g_b1 += deltaHidden

        cost /= len(batch)
        g_W /= len(batch)
        g_b1 /= len(batch)
        g_b2 /= len(batch)

        return cost, g_W, g_b1, g_b2

    def get_supcost_grad(self, batch, targets):

        batch_cost = 0.
        g_W = np.zeros(self.W.shape)
        g_b1 = np.zeros(self.b1.shape)
        g_Wlabel = np.zeros(self.Wlabel.shape)
        g_blabel = np.zeros(self.blabel.shape)

        for i, x in enumerate(batch):
            a = self._encode(x)
            h = self.actfunc.f(a)
            o = np.dot(self.Wlabel, h) + self.blabel
            (cost, out, grad) = self.OutModel.cost_out_grad(o, targets[i])
            batch_cost += cost

            deltaOut = grad
            g_Wlabel += np.outer(deltaOut, h)
            g_blabel += deltaOut

            deltaHidden = np.dot(self.Wlabel.T, deltaOut) * self.actfunc.df(a)
            g_W += np.outer(deltaHidden, x)
            g_b1 += deltaHidden

        batch_cost /= len(batch)
        g_W /= len(batch)
        g_b1 /= len(batch)
        g_Wlabel /= len(batch)
        g_blabel /= len(batch)

        return batch_cost, g_W, g_b1, g_Wlabel, g_blabel

    def train(self, data, epochs=2, batch_size=20, freeIndex=0):

        batch_num = len(data) / batch_size

        for epoch in xrange(epochs):
            total_cost = 0.
            self.eta *= 0.99
            for i in xrange(batch_num):
                batch = data[i*batch_size : (i+1)*batch_size]
                (cost, g_W, g_b1, g_b2) = self.get_cost_grad(batch)
                total_cost += cost
                self.W[freeIndex] -= self.eta * g_W[freeIndex]
                self.b1[freeIndex] -= self.eta * g_b1[freeIndex]
                self.b2 -= self.eta * g_b2

            print "Epoch: %d" % epoch
            print (1. / batch_num) * total_cost

    def trainSupervised(self, data, targets, epochs=10, batch_size=20):

        batch_num = len(data) / batch_size

        for epoch in xrange(epochs):
            total_cost = 0.
            self.eta = 0.99
            for batchIdx in xrange(batch_num):
                batch = data[batchIdx * batch_size : (batchIdx+1) * batch_size]
                batch_targets = targets[batchIdx * batch_size : (batchIdx+1) * batch_size]
                (cost, g_W, g_b1, g_Wlabel, g_blabel) = self.get_supcost_grad(batch, batch_targets)
                total_cost += cost
                self.W -= self.eta * g_W
                self.b1 -= self.eta * g_b1
                self.Wlabel -= self.eta * g_Wlabel
                self.blabel -= self.eta * g_blabel

            print "Epoch: %d" % epoch
            print (1. / batch_num) * total_cost

    def predict(self, data):

        if self.OutModel is None:
            return None

        predictedTargets = []
        predictedLabels = []

        for x in data:
            a = self._encode(x)
            h = self.actfunc.f(a)
            o = np.dot(self.Wlabel, h) + self.blabel
            pred = self.OutModel.p.f(o)
            idx = np.argmax(pred)
            predictedTargets.append(pred)
            predictedLabels.append(idx)

        return (predictedTargets, predictedLabels)

    def visualize_filters(self, name=None):
        tile_size = (int(np.sqrt(self.W[0].size)), int(np.sqrt(self.W[0].size)))
        panel_shape = (int(len(self.W)/11)+1, len(self.W) % 11)
        img = utils.visualize_weights(self.W, panel_shape, tile_size)
        if name is not None:
            img.save(name+".png", "PNG")
        img.show()


class ElasticLearning(object):
    def __init__(self, data, labels):
        self.data = data
        self.targets = labels
        self.visibles = len(data[0])

    def findCode(self, codeLimit=10):

        self.tmpW = None
        self.tmpb1 = None
        self.tmpb2 = None

        bitRange = range(1, codeLimit+1)

        for (idx, codeSize) in enumerate(bitRange):
            fixedRange = range(0, idx)
            print "CodeSize: %d/%d" % (codeSize, codeLimit)
            self.ae = Autoencoder(nvis=self.visibles, nhid=codeSize)
            if codeSize == 1:
                self.ae.train(self.data, epochs=2, freeIndex=idx)
                self.tmpW = self.ae.W
                self.tmpb1 = self.ae.b1
                self.tmpb2 = self.ae.b2
            else:
                self.ae.W[fixedRange] = self.tmpW
                self.ae.b1[fixedRange] = self.tmpb1
                self.ae.b2 = self.tmpb2
                self.ae.train(self.data, epochs=2, freeIndex=idx)
                self.tmpW = self.ae.W
                self.tmpb1 = self.ae.b1
                self.tmpb2 = self.ae.b2

        self.dump_model()

    def finetune(self):
        self.ae.init_supervised(len(self.targets[0]))
        self.ae.trainSupervised(self.data, self.targets, epochs=10, batch_size=20)

    def dump_model(self):
        cPickle.dump(self.ae, open(PATH_DATA_ROOT+r"/model.pkl", "w"))

    def load_model(self):
        self.ae = cPickle.load(open(PATH_DATA_ROOT+r"/model.pkl", "r"))
        return self.ae


class ClusterEvaluation(object):
    def __init__(self, data, model, codeLength):
        self.data = data
        self.model = model
        self.codeLength = codeLength

        self.clusters = {}

    def extractClustering(self):
        stats = []
        allUserIds = range(0, len(self.data)) # 6041
        userClusters = np.zeros((np.shape(self.data)[0], self.codeLength))
        for (i, d) in enumerate(self.data):
            code = self.model._encode(d)
            roundedCluster = [x for x in np.round(code)]
            userClusters[i] = roundedCluster
            roundedClusterStr = [str(int(x)) for x in roundedCluster]
            minCodeStr = "".join(roundedClusterStr)
            minUser = i
            d = np.linalg.norm(code - np.round(code))

            if minCodeStr not in self.clusters:
                self.clusters[minCodeStr] = {"userIds": [minUser], "userDists": [d]}
            else:
                self.clusters[minCodeStr]["userIds"].append(minUser)
                self.clusters[minCodeStr]["userDists"].append(d)

        codevecs = np.zeros((2**self.codeLength, self.codeLength))
        for i in xrange(2**self.codeLength):
            binStr = format(i, '00'+str(self.codeLength)+'b')
            codeVec = np.asarray([float(bit) for bit in binStr])
            codevecs[i] = codeVec
        recs = [self.model._decode(c) for c in codevecs]
        savemat(r"C:\Users\Johannes\Documents\EEXCESS\useritemmatrix\movielens\ml-1m\recs.mat", {"recs": recs})

    def printClustering(self, dataset):
        clusterSizes = []
        for c in self.clusters:
            mean = np.mean(np.asarray(self.clusters[c]["userDists"]))
            std = np.std(np.asarray(self.clusters[c]["userDists"]))
            clusterSizes.append(len(self.clusters[c]["userIds"]))
            print "%s : %d : mean %.3f : std %.3f :" % (c, len(self.clusters[c]["userIds"]), mean, std)
            # print "%s : %d : mean %.3f : std %.3f : %s" % (c, len(self.clusters[c]["userIds"]), mean, std, ", ".join([str(dataset.users.getUser(u)) for u in self.clusters[c]["userIds"]]))
            for (i,u) in enumerate(self.clusters[c]["userIds"]):
                if i < 20:
                    print str(dataset.users.getUser(u))
                else:
                    break



if __name__ == "__main__":

    import time

    train_data, test_data, valid_data = utils.load_mnist()
    K = 10

    print "Elastic Learning Algorithm:"
    ella = ElasticLearning(train_data[0], Utils.code1ofK(train_data[1], 10))
    strttime = time.time()
    ella.findCode()
    endtime = time.time()
    print "pre-training took %.2f s" % (endtime-strttime)
    ella.ae.visualize_filters("elastic-pretrained")
    print "Fine-Tuning with label information:"
    ella.finetune()
    ella.ae.visualize_filters("elastic-finetuned")
    print "Test Model:"
    (pred_targets, pred_labels) = ella.ae.predict(test_data[0])
    (pre, rec, f1, sup) = utils.validate(test_data[1], pred_labels)
    elastic = f1

    print "Composite Learning Algorithm:"
    ae = Autoencoder(nvis=784, nhid=10, eta=0.1)
    strttime = time.time()
    ae.train(train_data[0], epochs=20, batch_size=20, freeIndex=range(10))
    endtime = time.time()
    print "pre-training took %.2f s" % (endtime-strttime)
    ae.visualize_filters("compound-pretrained")
    print "Fine-Tuning with Label Information"
    ae.init_supervised(K)
    ae.trainSupervised(train_data[0], Utils.code1ofK(train_data[1], K), epochs=10, batch_size=20)
    ae.visualize_filters("compound-finetuned")
    print "Test Model:"
    (pred_targets, pred_labels) = ae.predict(test_data[0])
    (pre, rec, f1, sup) = utils.validate(test_data[1], pred_labels)
    compound = f1


    # ae = Autoencoder(nvis=784, nhid=10, eta=0.1)
    # ae.train(train_data[0], epochs=20, batch_size=20, freeIndex=range(10))
    # ae.visualize_filters()
    # ae.init_supervised(K)
    # train_data_targets = Utils.code1ofK(train_data[1], K)
    # ae.trainSupervised(train_data[0], train_data_targets, epochs=10, batch_size=500)

    import matplotlib.pyplot as plt
    n_groups = 10

    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.35

    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    rects1 = plt.bar(index, compound, bar_width,
                     alpha=opacity,
                     color='b',
                     error_kw=error_config,
                     label='Compound')

    rects2 = plt.bar(index + bar_width, elastic, bar_width,
                     alpha=opacity,
                     color='r',
                     error_kw=error_config,
                     label='Elastic')

    plt.xlabel('Digit Group')
    plt.ylabel('F1-Score')
    plt.title('F1-Scores on MNIST by Digit Groups and Method')
    plt.xticks(index + bar_width, ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'))
    plt.legend()

    plt.tight_layout()
    plt.show()