__author__ = 'johannesjurgovsky'

import time
from utils import utils
from utils.nnmath import *
import numpy as np
from numpy import random as rng
import theano
from config import Config
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from itertools import chain

class GraphView(object):
    def __init__(self, inputnode, outputnode):
        self.tiers = [[inputnode],[outputnode]]
        self._instantiated = False

    def instantiate(self):
        for tier in self.tiers:
            for n in tier:
                n.instantiate()
        self._instantiated = True

    def getDepth(self):
        return len(self.tiers)

    def getNodesInTier(self, tierIdx):
        return len(self.tiers[tierIdx])

    def insertTierAt(self, tierIdx):
        if tierIdx <= 0 or tierIdx > self.getDepth()-1 or self._instantiated:
            raise utils.GraphStructureError
        self.tiers.insert(tierIdx, [])

    def addToTier(self, tierIdx, node):
        if tierIdx <= 0 or tierIdx > self.getDepth()-2 or self._instantiated:
            raise utils.GraphStructureError
        self.tiers[tierIdx].append(node)
        node._setAsInputs(self.tiers[tierIdx-1])
        node._setAsOutputs(self.tiers[tierIdx+1])

    def __str__(self):
        return "\n---------\n".join(["\n".join([str(n) for n in tier]) for tier in self.tiers])


class ConnectivityMatrix(object):
    def __init__(self, file):
        cm = []
        for l in open(file, mode='r'):
            line = []
            for t in [x.rstrip() for x in l.split(';')]:
                b = 0
                if t == '1':
                    b = 1
                line.append(b)
            cm.append(line)
        self.np_cm = np.asarray(cm)

    def getMask(self):
        return self.np_cm

    def getSize(self):
        return self.np_cm.shape

    def showAsImage(self):
        imglist = [self.np_cm, np.invert(self.np_cm)]
        fig = plt.figure()

        im = plt.imshow(imglist[0], cmap=plt.get_cmap('gray_r'), vmin=0, vmax=1, interpolation='nearest')
        def init():
            im.set_data(imglist[0])

        def updatefig(i):
            a = im.get_array()
            # a= a * np.exp(-0.001*i)
            im.set_array(a)
            return [im]
        ani = animation.FuncAnimation(fig, updatefig, init_func=init, frames=100, interval=300, blit=False)
        plt.show()


class Graph(object):

    NODE_CNT = 0

    def __init__(self, inputs, outputs, connectivityMatrix, outfunc=T_OutFunc_Type.SOFTMAX, errfunc=T_ErrFunc_Type.CROSS_ENTROPY_MULTINOMIAL):

        self.input_dim = inputs
        self.output_dim = outputs

        self.outfunc = outfunc
        self.errfunc = errfunc

        self.connectivityMatrix = connectivityMatrix

        self._build()

    def _build(self):
        self.nodes = [InputNode(self.input_dim)] + [HiddenNode() for a in range(self.connectivityMatrix.getSize()[0]-2)] + [OutputNode(self.output_dim, outfunc=self.outfunc, errfunc=self.errfunc)]
        for (iIdx, row) in enumerate(self.connectivityMatrix.getMask()):
            for (oIdx, field) in enumerate(row):
                if iIdx < oIdx:
                    if field == 1:
                        self.nodes[iIdx].addOutputNode(self.nodes[oIdx])
        for n in self.nodes:
            n.instantiate()

    def propagate(self, x, y):
        evallist = []
        self.nodes[0]._inCache = x
        for (colIdx, col) in enumerate(self.connectivityMatrix.getMask().T):
            nodeIndices = [i for (i, f) in enumerate(col) if f == 1]
            for i in nodeIndices:
                if self.nodes[i] not in evallist:
                    self.nodes[i].propagate(x, y)
                    evallist.append(self.nodes[i])
        (out, r) = self.nodes[-1].propagate(x,y)
        print out
        return out


    def backpropagate(self, x, y):
        evallist = []
        dEdx = self.nodes[-1].backpropagate(x, y)
        for colIdx in xrange(self.connectivityMatrix.getSize()[1], 0, -1):# enumerate(reversed(self.connectivityMatrix.getMask().T)):
            nodeIndices = [i for (i, f) in enumerate(self.connectivityMatrix.getMask()[colIdx]) if f == 1]
            for i in nodeIndices:
                if self.nodes[i] not in evallist:
                    self.nodes[i].backpropagate(x, y)
                    evallist.append(self.nodes[i])
        (xgrad, r) = self.nodes[0].backpropagate(x, y)
        print xgrad
        return xgrad


    def __str__(self):
        return "\n".join([str(n) for n in self.nodes])


class T_Node_Type:
    INPUT_NODE = "input"
    HIDDEN_NODE = "hidden"
    OUTPUT_NODE = "output"


class Node(object):
    def __init__(self, nodeType):
        """
        A node in a computational graph. The node applies a function to its inputs and computes an output.
        The computed function is specific to the node type and is implemented in a particular subclass of this base class.
        :param nodeType: Identifier of this node.
        :return: A Node object.
        """
        Graph.NODE_CNT += 1
        self._id = Graph.NODE_CNT
        self._nodeType = nodeType
        self._inNodes = []
        self._inCache = []
        self._gradCache = []
        self._outNodes = []
        self._dim_in = 0
        self._dim_out = 1
        self._instantiated = False

    def addInputNode(self, node):
        node._outNodes.append(self)
        self._inNodes.append(node)
        self._dim_in += node._dim_out

    def addOutputNode(self, node):
        node._inNodes.append(self)
        node._dim_in += self._dim_out
        self._outNodes.append(node)

    def instantiate(self, weights=None, biases=None):
        raise NotImplementedError

    def f(self, x):
        """
        Computes the activation value of this node.
        :param x: An input vector as numpy array
        :return: Returns the activation value of this node
        """
        raise NotImplementedError

    def df(self, x, dEdo=None):
        """
        Computes dodx, the derivative of the node's output o with respect to all inputs x - or dEdx, if dEdo is given
        :param x: An input vector as numpy array
        :param dEdo: The derivative of the Error function E with respect to this node's output o
        :return: If dEdo is not None, returns the dEdo * dodx. Otherwise returns dodx.
        """
        raise NotImplementedError

    def _acceptOut(self, out, node):
        self._inCache[self._inNodes.index(node)] = out

    def _acceptGrad(self, grad, node):
        self._gradCache[self._outNodes.index(node)] = grad

    def propagate(self, x, y):
        missingNodeInputs = [self._inNodes[i] for i, n in enumerate(self._inCache) if n is None]
        if len(missingNodeInputs) == 0:
            x = np.hstack(self._inCache)
            out = self.f(x, y)
            print "Node %d : type = %s : input = %s : output = %s" % (self._id, self._nodeType, str(self._inCache), str(out))
            for n in self._outNodes:
                n._acceptOut(out, self)
        return (out, missingNodeInputs)

    def backpropagate(self, x, y):
        missingNodeOutputs = [self._outNodes[i] for i,n in enumerate(self._gradCache) if n is None]
        if len(missingNodeOutputs) == 0:
            x = np.hstack(self._inCache)
            dEdo = np.sum(np.hstack(self._gradCache))
            dEdx = self.df(x, dEdo)
            print "Node %d : type = %s : outputGrad = %s : inputGrad = %s" % (self._id, self._nodeType, str(self._gradCache), str(dEdx))
            for n in self._inNodes:
                n._acceptGrad(dEdx, self)
        return (dEdx, missingNodeOutputs)

    def __str__(self):
        return "Node %d (type=%s, nInNodes=%d, nOutNodes=%d, dim_in=%d, dim_out=%d)" % (self._id, self._nodeType, len(self._inNodes), len(self._outNodes), self._dim_in, self._dim_out)


class InputNode(Node):

    def __init__(self, nin):
        Node.__init__(self, T_Node_Type.INPUT_NODE)
        self._dim_in = nin
        self._dim_out = nin

    def addInputNode(self, node):
        raise utils.GraphStructureError

    def instantiate(self):
        self._instantiated = True
        self._gradCache = [None] * len(self._outNodes)

    def f(self, x, y):
        return x

    def df(self, x, y, dEdo=None):
        if dEdo is not None:
            return dEdo * np.ones(x.shape)
        else:
            return np.ones(x.shape)

    def __str__(self):
        return "%s" % (Node.__str__(self))


class HiddenNode(Node):

    def __init__(self, actfunc=actFuncs[T_Func_Type.TANH]):

        Node.__init__(self, T_Node_Type.HIDDEN_NODE)

        self.actfunc = actfunc

        self.w = None
        self.b = None

        self.cache = {}

    def instantiate(self, weights=None, bias=None):
        if weights is not None:
            assert weights.shape == (self._dim_in,)
            self.w = weights
        else:
            self.w = np.asarray(rng.uniform(low=-4 * np.sqrt(6. / (self._dim_out + self._dim_in)), high=4 * np.sqrt(6. / (self._dim_out + self._dim_in)), size=(self._dim_in,)), dtype=theano.config.floatX)

        if bias is not None:
            assert bias.shape == (self._dim_out,)
            self.b = bias
        else:
            self.b = np.zeros(shape=(self._dim_out,), dtype=theano.config.floatX)
        self._inCache = [None] * len(self._inNodes)
        self._gradCache = [None] * len(self._outNodes)
        self._instantiated = True


    def f(self, x, y):
        a = np.dot(self.w, x) + self.b
        o = self.actfunc.f(a)
        self.cache[hash(x.tostring())] = (a, o, None)
        return o

    def df(self, x, y, dEdo=None):
        hx = hash(x.tostring())
        if hx not in self.cache:
            self.f(x)
        (a, o, old_dodx) = self.cache[hx]
        dodx = self.actfunc.df(a) * self.w
        self.cache[hx] = (a, o, dodx)
        if dEdo is not None:
            return dEdo * dodx
        else:
            return dodx

    def __str__(self):
        return "%s\nw = %s\nb = %s\ncache = %s" % (Node.__str__(self), self.w, self.b, self.cache)


class OutputNode(Node):

    def __init__(self, nout, outfunc=T_OutFunc_Type.SOFTMAX, errfunc=T_ErrFunc_Type.CROSS_ENTROPY_MULTINOMIAL):
        Node.__init__(self, T_Node_Type.OUTPUT_NODE)
        self._dim_out = nout
        self.W = None
        self.b = None
        self.OutModel = OutputModel(outfunc, errfunc)

    def addOutputNode(self, node):
        raise utils.GraphStructureError

    def instantiate(self, weights=None, biases=None):
        if weights is not None:
            assert weights.shape == (self._dim_out, self._dim_in)
            self.W = weights
        else:
            self.W = np.asarray(rng.uniform(low=-4 * np.sqrt(6. / (self._dim_out + self._dim_in)), high=4 * np.sqrt(6. / (self._dim_out + self._dim_in)), size=(self._dim_out, self._dim_in)), dtype=theano.config.floatX)

        if biases is not None:
            assert biases.shape == (self._dim_out,)
            self.b = biases
        else:
            self.b = np.zeros(shape=(self._dim_out,), dtype=theano.config.floatX)
        self._inCache = [None] * len(self._inNodes)
        self._gradCache = [None] * len(self._outNodes)
        self._instantiated = True

    def f(self, x, y):
        a = np.dot(self.W, x) + self.b
        (cost, o, dEda) = self.OutModel.cost_out_grad(a, y)
        return (cost, o)

    def df(self, x, y):
        a = np.dot(self.W, x) + self.b
        (cost, o, dEda) = self.OutModel.cost_out_grad(a, y)
        dEdx = np.dot(self.W.T, dEda)
        return dEdx

    def __str__(self):
        return "%s\nOutput Model = %s\nW = %s\nb = %s" % (Node.__str__(self), self.OutModel, self.W, self.b)


if __name__ == "__main__":
    x = np.asarray([-1.0, 1.0, 0.5])
    y = np.asarray([1., 0.])
    cm = ConnectivityMatrix(utils.get_full_path(Config.PATH_DATA_ROOT, Config.CONNECTIVITY_FILE))
    g = Graph(3, 2, cm)

    print g
    print "propagation"
    g.propagate(x, y)
    cm.showAsImage()
    g.backpropagate(x, y)