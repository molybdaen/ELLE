__author__ = 'Johannes'

from config import Config
import numpy as np
import theano
from numpy import random as rng
from utils import utils, nnmath
import cPickle


class Autoencoder(object):
    def __init__(self, nvis=100, nhid=50, eta=0.1, actfunc=nnmath.actFuncs[nnmath.T_Func_Type.TANH]):

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

    def init_supervised(self, nout):
        self.output_size = nout
        self.Wlabel = np.asarray(rng.uniform(low=-4 * np.sqrt(6. / (self.output_size + self.hidden_size)), high=4 * np.sqrt(6. / (self.output_size + self.hidden_size)), size=(self.output_size, self.hidden_size)), dtype=theano.config.floatX)
        self.blabel = np.zeros(shape=(self.output_size,), dtype=theano.config.floatX)
        self.OutModel = nnmath.OutputModel(nnmath.T_OutFunc_Type.SOFTMAX, nnmath.T_ErrFunc_Type.CROSS_ENTROPY_MULTINOMIAL)

    def get_cost_grad(self, batch):

        cost = 0.
        g_W = np.zeros(self.W.shape)
        g_b1 = np.zeros(self.b1.shape)
        g_b2 = np.zeros(self.b2.shape)

        for x in batch:
            a = self._encode(x)
            h = self.actfunc.f(a)
            p = self._decode(h)

            cost += nnmath.rec_error(p, x)

            deltaOut = nnmath.drec_error(p, x)

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

        nFilters = len(self.W)
        nFeats = self.W.shape[1]

        tile_size = (int(np.sqrt(nFeats)), int(np.sqrt(nFeats)))
        panel_shape = (1, nFilters)
        img = utils.visualize_weights(self.W, panel_shape, tile_size)
        if name is not None:
            img.save(utils.get_full_path(Config.PATH_DATA_ROOT, name + ".png"), "PNG")
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
                self.ae.train(self.data, epochs=1, freeIndex=idx)
                self.tmpW = self.ae.W
                self.tmpb1 = self.ae.b1
                self.tmpb2 = self.ae.b2
            else:
                self.ae.W[fixedRange] = self.tmpW
                self.ae.b1[fixedRange] = self.tmpb1
                self.ae.b2 = self.tmpb2
                self.ae.train(self.data, epochs=1, freeIndex=idx)
                self.tmpW = self.ae.W
                self.tmpb1 = self.ae.b1
                self.tmpb2 = self.ae.b2

        self.dump_model()

    def finetune(self):
        self.ae.init_supervised(len(self.targets[0]))
        self.ae.trainSupervised(self.data, self.targets, epochs=10, batch_size=20)

    def dump_model(self):
        cPickle.dump(self.ae, open(utils.get_full_path(Config.PATH_DATA_ROOT, r"model.pkl"), "w"))

    def load_model(self):
        self.ae = cPickle.load(open(utils.get_full_path(Config.PATH_DATA_ROOT, r"model.pkl"), "r"))
        return self.ae


if __name__ == "__main__":

    import time

    train_data, test_data, valid_data = utils.load_mnist()

    K = 10

    codeLimit = 3

    print "Elastic Learning Algorithm:"
    ella = ElasticLearning(train_data[0], utils.code1ofK(train_data[1], K))
    strttime = time.time()
    ella.findCode(codeLimit)
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
    ae = Autoencoder(nvis=784, nhid=codeLimit, eta=0.1)
    strttime = time.time()
    ae.train(train_data[0], epochs=20, batch_size=20, freeIndex=range(codeLimit))
    endtime = time.time()
    print "pre-training took %.2f s" % (endtime-strttime)
    ae.visualize_filters("compound-pretrained")
    print "Fine-Tuning with Label Information"
    ae.init_supervised(K)
    ae.trainSupervised(train_data[0], utils.code1ofK(train_data[1], K), epochs=10, batch_size=20)
    ae.visualize_filters("compound-finetuned")
    print "Test Model:"
    (pred_targets, pred_labels) = ae.predict(test_data[0])
    (pre, rec, f1, sup) = utils.validate(test_data[1], pred_labels)
    compound = f1



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