__author__ = 'johannesjurgovsky'

import time
import numpy as np
import cPickle
from utils import nnmath, utils
from config import Config

DATA = ()


def func():
    return DATA[0], DATA[1], 10


class Autoencoder(object):

    _MAX_EPOCHS = 500
    _MAX_EPOCHS_SUPERVISED = 50

    STR_ERRORS = "errors"
    STR_TIMES = "times"
    STR_SCORES = "scores"

    def __init__(self, nvis, nhid, eta=0.1, caching=True, log_level=1):

        self.visible_size = nvis
        self.hidden_size = nhid

        weight_scaling = 1. / self.visible_size
        self.W = weight_scaling * (np.random.random((self.hidden_size, self.visible_size)) * 2. - 1.)
        self.b1 = np.zeros(shape=(self.hidden_size,))
        self.b2 = np.zeros(shape=(self.visible_size,))

        self.eta = eta
        self.caching = caching
        self.log_level = log_level

    def _sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def _sigmoid_prime(self, y):
        return y * (1. - y)

    def _encode(self, x, fwd_mask):
        return self._sigmoid(np.dot(self.W[fwd_mask], x) + self.b1[fwd_mask])

    def _decode(self, h, fwd_mask):
        return self._sigmoid(np.dot(self.W[fwd_mask].T, h) + self.b2)

    def _sum_squared(self, x, z):
        return np.sum((x-z)**2.)

    def _sum_squared_prime(self, x, z):
        return -2. * (x - z)

    def _init_cache(self, size):
        # Cache for input values of output layer
        self.output_cache = np.zeros((size, self.visible_size))

    def _add_to_cache(self, data, fwd_mask):
        for (i, x) in enumerate(data):
            y = self._encode(x, fwd_mask)
            in_z = np.dot(self.W[fwd_mask].T, y)
            self.output_cache[i] += in_z

    def _init_supervised(self, nout, fwd_mask):
        width = len(fwd_mask[fwd_mask])
        self.output_size = nout
        weight_scaling = 1. / width
        self.Wlabel = weight_scaling * (np.random.random((self.output_size, width)) * 2. - 1.)
        self.blabel = np.zeros(shape=(self.output_size,))
        self.OutModel = nnmath.OutputModel(nnmath.T_OutFunc_Type.SOFTMAX, nnmath.T_ErrFunc_Type.CROSS_ENTROPY_MULTINOMIAL)

    def _cost_grad_sup(self, batch, targets, fwd_mask):

        batch_err = 0.
        g_Wlabel = np.zeros(self.Wlabel.shape)
        g_blabel = np.zeros(self.blabel.shape)

        for i, x in enumerate(batch):
            y = self._encode(x, fwd_mask)
            o = np.dot(self.Wlabel, y) + self.blabel

            (err, out, deltaOut) = self.OutModel.cost_out_grad(o, targets[i])

            batch_err += err

            g_Wlabel += np.outer(deltaOut, y)
            g_blabel += deltaOut

        batch_err /= len(batch)
        g_Wlabel /= len(batch)
        g_blabel /= len(batch)

        return batch_err, g_Wlabel, g_blabel

    def _cost_grad(self, batch, fwd_mask, err_mask, batchIdx):

        batch_size = float(len(batch))
        batch_error = 0.
        g_W = np.zeros(self.W[err_mask].shape)
        g_b1 = np.zeros(self.b1[err_mask].shape)
        g_b2 = np.zeros(self.b2.shape)

        relative_err_mask = np.logical_and(fwd_mask[fwd_mask], err_mask[fwd_mask])

        for (i, x) in enumerate(batch):

            y = self._encode(x, fwd_mask)
            # z = self._decode(y, fwd_mask)
            in_z = np.dot(self.W[fwd_mask].T, y) + self.b2
            if self.caching:
                in_z += self.output_cache[batchIdx+i]
            z = self._sigmoid(in_z)
            err = self._sum_squared(x, z)

            batch_error += err

            deltaOut = self._sum_squared_prime(x, z) * self._sigmoid_prime(z)
            r = np.outer(y[relative_err_mask], deltaOut)
            g_W += r
            g_b2 += deltaOut

            deltaHidden = np.dot(self.W[err_mask], deltaOut) * self._sigmoid_prime(y)

            g_W += np.outer(deltaHidden[relative_err_mask], x)
            g_b1 += deltaHidden[relative_err_mask]

        batch_error /= batch_size
        g_W /= batch_size
        g_b1 /= batch_size
        g_b2 /= batch_size

        return batch_error, g_W, g_b1, g_b2

    def train_supervised(self, data, targets, outputs, fwd_mask, epochs=None, mini_batch_size=None, stop_err_delta=0.1):

        self._init_supervised(outputs, fwd_mask)

        if mini_batch_size is None:
            mini_batch_size = len(data)

        num_batches = len(data) / mini_batch_size
        max_epochs = epochs
        # fwd_mask = np.ones(self.hidden_size, dtype=bool)
        errors = []

        if epochs is None:
            max_epochs = Autoencoder._MAX_EPOCHS_SUPERVISED

        if self.log_level > 0: print "Supervised training with Cross-Entropy error"

        prev_error = 10e8

        for e in xrange(max_epochs):
            total_error = 0.

            for b in xrange(num_batches):
                batch = data[b * mini_batch_size : (b + 1) * mini_batch_size]
                batch_targets = targets[b * mini_batch_size : (b + 1) * mini_batch_size]
                (err, g_Wlabel, g_blabel) = self._cost_grad_sup(batch, batch_targets, fwd_mask)
                total_error += err
                self.Wlabel -= self.eta * g_Wlabel
                self.blabel -= self.eta * g_blabel

            curr_error = (1. / num_batches) * total_error
            errors.append(curr_error)

            err_delta = prev_error / curr_error

            if self.log_level > 1: print "Epoch: %d, CEE: %.5f" % (e, curr_error)
            if self.log_level > 2: print "ErrorDelta: %.5f" % err_delta

            if epochs is None and err_delta < (1. + stop_err_delta):
                break

            prev_error = curr_error

        return errors

    def train(self, data, elastic=True, epochs=None, mini_batch_size=None, stop_err_delta=0.1, supervisedDataCallback=func):
        """
        Train an autoencoder model on a dataset. You can specify whether all hidden nodes should be trained
        simultaneously or one after another.

        :param data: The dataset as a 2D numpy array where each row is a single example
        :param elastic: A boolean flag indicating whether the hidden nodes are trained consecutively or simultaneously
        :param epochs: An Integer value indicating the number of training passes over the dataset. If ``epochs`` is
                        set to ``None``, training proceeds until the error drops by less than ``stop_err_delta``
                        relative to the error in the previous epoch.
        :param mini_batch_size: An Integer value indicating the number of examples to process before taking a gradient step.
                        If ``mini_batch_size`` is set to ``None``, batch gradient descent is performed. If set to ``1``,
                        stochastic gradient descent is performed. Any other number corresponds to gradient descent
                        with mini-batches.
        :param stop_err_delta: The fraction of error drop between two epochs. ``stop_err_delta`` * 100 is the
                        percentage of change. You can specify a hard upper bound for the maximum number of training
                        epochs by setting ``Autoencoder._MAX_EPOCHS``.
        :return: A list of epoch errors
        """

        if mini_batch_size is None:
            mini_batch_size = len(data)

        num_batches = len(data) / mini_batch_size
        max_epochs = epochs
        node_indices = np.arange(0, self.hidden_size, dtype=int)
        mask = np.ones(self.hidden_size, dtype=bool)
        nodes_to_train = [node_indices[mask]]
        logs = {}

        if elastic:
            nodes_to_train = [[idx] for idx in node_indices[mask]]
            if self.caching:
                self._init_cache(len(data))

        if epochs is None:
            max_epochs = Autoencoder._MAX_EPOCHS

        if self.log_level > 0: print "Unsupervised training with reconstruction error"

        for h in nodes_to_train:
            # Define masks for forward and backward pass. Only nodes at indices masked with 1 are computed.
            # TODO: an adjacency matrix might be a better way to represent the masks.
            # TODO: Heuristics might be found that dynamically determine which nodes should participate in forward and backward pass
            fwd_mask = mask
            err_mask = mask
            if elastic:
                fwd_mask = np.zeros(self.hidden_size, dtype=bool)
                if self.caching:
                    fwd_mask[h[0]] = True
                else:
                    fwd_mask[node_indices<=h[0]] = True
                err_mask = np.zeros(self.hidden_size, dtype=bool)
                err_mask[h] = True

            if self.log_level > 0: print "Train node(s): %s" % (str(h))
            if self.log_level > 1: print "Forward mask:\n%s\nBackward mask:\n%s" % (str(fwd_mask.astype(int)), str(err_mask.astype(int)))

            prev_error = 10e8
            logs[str(h[-1])] = {}
            node_errors = []
            node_times = []

            for e in xrange(max_epochs):
                total_error = 0.
                strttime = time.time()

                for b in xrange(num_batches):
                    batch = data[b * mini_batch_size: (b + 1) * mini_batch_size]
                    (err, g_W, g_b1, g_b2) = self._cost_grad(batch, fwd_mask, err_mask, b*mini_batch_size)
                    total_error += err
                    self.W[err_mask] -= self.eta * g_W
                    self.b1[err_mask] -= self.eta * g_b1
                    self.b2 -= self.eta * g_b2

                curr_error = (1. / num_batches) * total_error

                node_errors.append(curr_error)
                stoptime = time.time()
                node_times.append((stoptime-strttime))

                err_delta = prev_error / curr_error

                if self.log_level > 1: print "Epoch: %d, SSE: %.5f" % (e, curr_error)
                if self.log_level > 2: print "ErrorDelta: %.5f" % err_delta

                if epochs is None and err_delta < (1. + stop_err_delta):
                    break

                prev_error = curr_error

            logs[str(h[-1])][Autoencoder.STR_ERRORS] = node_errors
            logs[str(h[-1])][Autoencoder.STR_TIMES] = node_times

            if supervisedDataCallback is not None:
                if h[-1] == 0 or h[-1] % 10 == 9:
                    traindata, testdata, K = supervisedDataCallback()
                    sup_fwd_mask = np.zeros(self.hidden_size, dtype=bool)
                    sup_fwd_mask[node_indices<=h[-1]] = True
                    if self.log_level > 2: print "Forward mask:\n%s\nBackward mask:\n%s" % (str(sup_fwd_mask.astype(int)), str(sup_fwd_mask.astype(int)))
                    self.train_supervised(traindata[0], utils.code1ofK(traindata[1], K), K, sup_fwd_mask, epochs=None, mini_batch_size=20, stop_err_delta=0.001)
                    (precision, recall, f1) = self.test(testdata[0], testdata[1], K, sup_fwd_mask)
                    logs[str(h[-1])][Autoencoder.STR_SCORES] = (precision, recall, f1)

            if elastic and self.caching:
                self._add_to_cache(data, fwd_mask)

        return logs

    def predict(self, data, fwd_mask):

        if self.OutModel is None:
            return None

        predictedTargets = []
        predictedLabels = []

        for x in data:
            y = self._encode(x, fwd_mask)
            o = np.dot(self.Wlabel, y) + self.blabel
            pred = self.OutModel.p.f(o)
            idx = np.argmax(pred)
            predictedTargets.append(pred)
            predictedLabels.append(idx)

        return (predictedTargets, predictedLabels)

    def encode_all(self, data):
        return np.asarray([self._encode(x, np.ones(self.hidden_size, dtype=bool)) for x in data])

    def decode_all(self, codes):
        return np.asarray([self._decode(code, np.ones(self.hidden_size, dtype=bool)) for code in codes])

    def test(self, test_data, test_labels, K, fwd_mask):

        if self.log_level > 0: print "Test model"

        (pred_targets, pred_labels) = self.predict(test_data, fwd_mask)
        (precision, recall, f1, sup) = utils.validate(test_labels, pred_labels, K)

        if self.log_level > 0:
            print precision
            print recall
            print f1

        return precision, recall, f1

    def visualize_filters(self, mode, panel_shape=None, filter_shape=None, name=None):
        """
        Plot weight matrix in a grid of shape ``panel_shape`` with each cell showing the weights of a hidden node,
        shaped as given by ``filter_shape``.
        :param mode: Either "RGB" or "GREY"
        :param panel_shape:
        :param filter_shape:
        :param name:
        :return:
        """
        num_filters = self.W.shape[0]
        num_features = self.W.shape[1]

        if panel_shape is None:
            panel_shape = (np.ceil(num_filters / 10.).astype(int), 10 + num_filters % 10)

        if filter_shape is None:
            filter_shape = (int(np.sqrt(num_features)), int(np.sqrt(num_features)))

        img = utils.visualize_weights(self.W, panel_shape, filter_shape, mode)

        if name is not None:
            img.save(utils.get_full_path(Config.PATH_EVAL_ROOT, name + "-filters.png"), "PNG")
        img.show()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['output_cache'] = np.asarray([])
        return state

    def __setstate__(self, newstate):
        self.__dict__.update(newstate)

    def dump_model(self, name):
        cPickle.dump(self, open(utils.get_full_path(Config.PATH_EVAL_ROOT, r"%s.pkl" % name), "w"))

    @staticmethod
    def load_model(name):
        return cPickle.load(open(utils.get_full_path(Config.PATH_EVAL_ROOT, r"%s.pkl" % name), "r"))

if __name__ == "__main__":

    K = 10
    hidden_size = 200
    data_name = "mnist"
    mode_name = "compound"
    id = (mode_name + "-" + data_name + "-" + str(hidden_size))

    train_data, test_data, valid_data = utils.load_mnist()
    DATA = (train_data, test_data, valid_data)

    # # train_data, test_data = utils.load_cifar()
    # # user_attribute_matrix, user_item_matrix = utils.load_movielens()

    # def test_elastic(hidden_size):
    #     ae = Autoencoder(nvis=len(train_data[0][0]), nhid=hidden_size, log_level=1, caching=True)
    #     strt = time.time()
    #     logs = ae.train(train_data[0], elastic=True, epochs=None, mini_batch_size=20, stop_err_delta=0.01, supervisedDataCallback=func)
    #     end = time.time()
    #     print "Total Training Time: %.3f" % (end-strt)
    #     ae.visualize_filters("GREY", name=id)
    #     return logs
    #
    def test_compound(hidden_size):
        total_logs = {}
        for i in range(0, hidden_size, 10):
            s = i+1
            ae = Autoencoder(nvis=len(train_data[0][0]), nhid=s, log_level=1, caching=False)
            strt = time.time()
            logs = ae.train(train_data[0], elastic=False, epochs=None, mini_batch_size=20, stop_err_delta=0.01)
            end = time.time()
            print "Total Training Time: %.3f" % (end-strt)
            total_logs[str(i)] = logs[str(i)]
            ae.visualize_filters("GREY", name=id)
        return total_logs

    # e_logs = test_elastic(hidden_size)
    # cPickle.dump(e_logs, open(utils.get_full_path(Config.PATH_EVAL_ROOT, "results-%s.pkl" % id), 'w'))

    c_logs = test_compound(hidden_size)
    cPickle.dump(c_logs, open(utils.get_full_path(Config.PATH_EVAL_ROOT, "results-%s.pkl" % id), 'w'))

    # e_logs = cPickle.load(open(utils.get_full_path(Config.PATH_EVAL_ROOT, "results-elastic.pkl"), 'r'))
    # c_logs = cPickle.load(open(utils.get_full_path(Config.PATH_EVAL_ROOT, "results-compound.pkl"), 'r'))

    # from viz.plotty import Plotty
    #
    # p = Plotty(None, None)
    # p.compareElasticCompound(e_logs, e_logs)