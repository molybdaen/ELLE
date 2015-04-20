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

PATH_DATA_ROOT = r"C:\Users\Johannes\Documents\EEXCESS\useritemmatrix\movielens\ml-1m"


class MLib:
    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def dlinear(x):
        return np.ones(x.shape)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def dtanh(x):
        tan = MLib.tanh(x)
        return (1. - tan**2.)

    @staticmethod
    def sigmoid(x):
        return 1. / (1. + np.exp(-x))

    @staticmethod
    def dsigmoid(x):
        sig = MLib.sigmoid(x)
        return sig*(1.-sig)

    @staticmethod
    def softmax(x):
        e = np.exp(x)
        return e / np.sum(e)#[:, np.newaxis]

    @staticmethod
    def dsoftmax(x):
        """
        NOTE: When computing the gradient of a combination of softmax output and crossentropy error,
        use :func:`~slowlearning.dsce_error` instead.
        Computing :func:`~slowlearning.dce_error` and :func:`~slowlearning.dsoftmax` separately is computationally very inefficient
        :param x:
        :return: Jacobean matrix. partial derivatives of all outputs :func:`~slowlearning.softmax` with respect to all inputs x
        """
        p = MLib.softmax(x)
        Ds = -np.outer(p, p)
        di = np.diag_indices(len(x))
        Ds[di] = p-p**2.
        return Ds

    @staticmethod
    def rec_error(p, y):
        return 0.5 * np.sum((p - y)**2.)

    @staticmethod
    def drec_error(p, y):
        return (p - y)

    @staticmethod
    def dlinrec_error(x, y):
        return x - y

    @staticmethod
    def ceb_error(p, y):
        eps = 1e-10
        return - np.sum(y * np.log(p + eps) + (1. - y) * np.log(1. - p + eps))

    @staticmethod
    def dceb_error(p, y):
        return - y * 1. / p + (1. - y) / (1. - p)

    @staticmethod
    def dlogceb_error(x, y):
        p = MLib.sigmoid(x)
        return - y * (1. - p) + (1. - y) * p

    @staticmethod
    def cem_error(p, y):
        eps = 1e-10
        return - np.sum(y * np.log(p + eps))

    @staticmethod
    def dcem_error(p, y):
        return - y * 1. / p

    @staticmethod
    def dsofcem_error(x, y):
        return MLib.softmax(x)-y


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

actFuncs = {T_Func_Type.TANH: Function(MLib.tanh, MLib.dtanh),
            T_Func_Type.LOGISTIC: Function(MLib.sigmoid, MLib.dsigmoid)}

outFuncs = {T_OutFunc_Type.SOFTMAX: Function(MLib.softmax, MLib.dsoftmax),
            T_OutFunc_Type.LINEAR: Function(MLib.linear, MLib.dlinear),
            T_OutFunc_Type.LOGISTIC: Function(MLib.sigmoid, MLib.dsigmoid)}

errFuncs = {T_ErrFunc_Type.RECONSTRUCTION: Function(MLib.rec_error, MLib.drec_error),
            T_ErrFunc_Type.CROSS_ENTROPY_BINOMIAL: Function(MLib.ceb_error, MLib.dceb_error),
            T_ErrFunc_Type.CROSS_ENTROPY_MULTINOMIAL: Function(MLib.cem_error, MLib.dcem_error)}


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
        self.p = outFuncs[outFuncType].f
        self.E = errFuncs[errFuncType].f
        if outFuncType == T_OutFunc_Type.SOFTMAX and errFuncType == T_ErrFunc_Type.CROSS_ENTROPY_MULTINOMIAL:
            self.dEdx = MLib.dsofcem_error
        elif outFuncType == T_OutFunc_Type.LOGISTIC and errFuncType == T_ErrFunc_Type.CROSS_ENTROPY_BINOMIAL:
            self.dEdx = MLib.dlogceb_error
        elif outFuncType == T_OutFunc_Type.LINEAR and errFuncType == T_ErrFunc_Type.RECONSTRUCTION:
            self.dEdx = MLib.dlinrec_error
        else:
            self.dEdx = lambda x, y: errFuncs[errFuncType].df(outFuncs[outFuncType].f(x), y) * outFuncs[outFuncType].df(x)

    def cost_out_grad(self, x, y):
        out = self.p(x)
        return (self.E(out, y), out, self.dEdx(x, y))


class Autoencoder():
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

        batchstel = 1. / len(batch[0])

        for x in batch:
            a = self._encode(x)
            h = self.actfunc.f(a)
            p = self._decode(h)

            cost += batchstel * MLib.rec_error(p, x)

            deltaOut = batchstel * MLib.drec_error(p, x)

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

    def predict(self, data, labels):
        digitdata = []
        digitlabel = []
        for digit in range(0,10):
            digitdat = [self._encode(x) for x in data[labels==digit]]
            digitlab = labels[labels==digit]
            digitdata.append(np.vstack(digitdat[0:20]))
            digitlabel.append(np.vstack(digitlab[0:20]))

        npstack = np.vstack(digitdata)
        nplabs = np.vstack(digitlabel)
        savemat(r"C:\Users\Johannes\Documents\EEXCESS\useritemmatrix\movielens\ml-1m\mnist-codes.mat", {"mnistcodes": npstack})
        savemat(r"C:\Users\Johannes\Documents\EEXCESS\useritemmatrix\movielens\ml-1m\mnist-labels.mat", {"mnistlabels": nplabs})

    def visualize_weights(self, W, pixwidth=1, ax=None, grayscale=True):
        savemat(r"C:\Users\Johannes\Documents\EEXCESS\useritemmatrix\movielens\ml-1m\W.mat", {"W": W})

        (N, M) = W.shape
        tN = int(np.sqrt(N))
        tM = int(np.sqrt(M))
        W = np.reshape(W.flatten(), (28, 28))
        (N, M) = W.shape
        # Need to create a new Axes?
        if(ax == None):
            ax = P.figure().gca()
        # extents = Left Right Bottom Top
        exts = (0, pixwidth * M, 0, pixwidth * N)
        if(grayscale):
            ax.imshow(W,
                      interpolation='nearest',
                      cmap=CM.gray,
                      extent=exts)
        else:
            ax.imshow(W,
                      interpolation='nearest',
                      extent=exts)

        ax.xaxis.set_major_locator(MT.NullLocator())
        ax.yaxis.set_major_locator(MT.NullLocator())
        P.show()

    def visualize_filters(self):
        tile_size = (int(np.sqrt(self.W[0].size)), int(np.sqrt(self.W[0].size)))
        panel_shape = (int(len(self.W)/11)+1, len(self.W) % 11)
        img = utils.visualize_weights(self.W, panel_shape, tile_size)
        img.show()

    def visualize_data(self, data):
        tile_size = (int(np.sqrt(data[0].size)), int(np.sqrt(data[0].size)))
        panel_shape = (int(len(data)/11)+1, len(data) % 11)
        img = utils.visualize_weights(data, panel_shape, tile_size)
        img.show()


class ElasticLearning(object):
    def __init__(self, data, dataset):
        self.uamat = data
        self.visibles = self.uamat.shape[1]
        self.dataset = dataset

    def findCode(self, codeLimit=10):
        self.tmpW = None
        self.tmpb1 = None
        self.tmpb2 = None
        for i in xrange(1,codeLimit):
            print "CodeSize: %d/%d" % (i, codeLimit)
            self.ae = Autoencoder(nvis=self.visibles, nhid=i)
            if i == 1:
                self.ae.train(self.uamat, epochs=3, freeIndex=i-1)
                self.tmpW = self.ae.W
                self.tmpb1 = self.ae.b1
                self.tmpb2 = self.ae.b2
                # self.ae.visualize_filters()
            else:
                self.ae.W[0:i-1] = self.tmpW
                self.ae.b1[0:i-1] = self.tmpb1
                self.ae.b2 = self.tmpb2
                self.ae.train(self.uamat, epochs=3, freeIndex=i-1)
                self.tmpW = self.ae.W
                self.tmpb1 = self.ae.b1
                self.tmpb2 = self.ae.b2
                # self.ae.visualize_filters()

        # self.ae.visualize_filters()
        self.dump_model()
        # self.ae = Autoencoder(self.visibles, codeLimit-1)
        # self.ae.W = self.tmpW
        # self.ae.b1 = self.tmpb1
        # self.ae.b2 = self.tmpb2
        # self.ae.train(self.uamat, epochs=10, batch_size=20, freeIndex=range(0,codeLimit-1))
        # self.ae.visualize_filters()

    def dump_model(self):
        cPickle.dump(self.ae, open(PATH_DATA_ROOT+r"\model.pkl", "w"))

    def load_model(self):
        self.ae = cPickle.load(open(PATH_DATA_ROOT+r"\model.pkl", "r"))
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
    # data = MovielensDataset(PATH_DATA_ROOT)
    # uamat = data.getUsersAttributesMatrix()
    # print uamat.shape
    # ae = Autoencoder(30, 9)
    # ae.train(uamat, epochs=55, freeIndex=range(0,9))
    # ae.visualize_weights()

    # codeLength = 7
    #
    # data = MovielensDataset(PATH_DATA_ROOT)
    # uamat = data.getUsersAttributesMatrix()


    train_data, test_data, valid_data = utils.load_mnist()
    K = 10
    ae = Autoencoder(nvis=784, nhid=100, eta=0.1)
    ae.train(train_data[0], epochs=10, batch_size=500, freeIndex=range(100))
    ae.init_supervised(K)
    train_data_targets = Utils.code1ofK(train_data[1], K)
    ae.trainSupervised(train_data[0], train_data_targets, epochs=10, batch_size=500)

    #
    # el = ElasticLearning(uamat, data)
    # el.findCode(codeLength)
    # ae = el.load_model()
    # ce = ClusterEvaluation(uamat, ae, codeLength-1)
    # ce.extractClustering()
    # ce.printClustering(data)


    # a = np.asarray([0.99, 0.01, 1.01, 0.00])
    # y = np.asarray([1.0, 0.0, 0.0, 0.0])
    # print MLib.softmax(a)
    # print np.sum(MLib.softmax(a))
    # print MLib.dsoftmax(a)
    #
    # om = OutputModel(T_OutFunc_Type.SOFTMAX, T_ErrFunc_Type.CROSS_ENTROPY_MULTINOMIAL)
    # print om.cost_out_grad(a, y)
    # print np.dot(MLib.dcem_error(MLib.softmax(a), y), MLib.dsoftmax(a))
    # print MLib.dsofcem_error(a, y)
    #
    # om = OutputModel(T_OutFunc_Type.LOGISTIC, T_ErrFunc_Type.CROSS_ENTROPY_BINOMIAL)
    # print om.cost_out_grad(a, y)
    # print MLib.dceb_error(MLib.sigmoid(a), y) * MLib.dsigmoid(a)
    # print MLib.dlogceb_error(a, y)
    #
    # om = OutputModel(T_OutFunc_Type.LINEAR, T_ErrFunc_Type.RECONSTRUCTION)
    # print om.cost_out_grad(a, y)
    # print MLib.drec_error(MLib.linear(a), y) * MLib.dlinear(a)
    # print MLib.dlinrec_error(a, y)



    # ce = ClusterEvaluation(uamat, ae, codeLength)
    # ce.extractClustering()
    # ce.printClustering(data)

    # listDigits = []
    # for (i, x) in enumerate(train_data[0]):
    #     print train_data[1][i]
    #     rec = ae._decode(ae._encode(x))
    #     listDigits.append(x)
    #     listDigits.append(rec)
    #     if i > 47:
    #         break
    # ae.visualize_data(np.vstack(listDigits))
    # ae.visualize_filters()
    # ae.predict(train_data[0], train_data[1])
    # el.findCode(10)