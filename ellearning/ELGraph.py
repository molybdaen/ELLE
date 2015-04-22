__author__ = 'johannesjurgovsky'

from utils import utils
from utils.nnmath import *
import numpy as np
from numpy import random as rng
import theano


class Graph(object):

    NODE_CNT = 0
    NODES = []

    def __init__(self, data):
        pass


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
        self._outNodes = []
        self._nInputs = 0
        self._nOutputs = 0
        self._dim_in = 0
        self._dim_out = 1
        self._instantiated = False

    def _setInNodes(self, nodes):
        self._inNodes = []
        for n in nodes:
            self._inNodes.append(n)
            self._dim_in += n._dim_out
        self._nInputs = len(nodes)

    def _setOutNodes(self, nodes):
        self._outNodes = []
        for n in nodes:
            self._outNodes.append(n)
        self._nOutputs = len(nodes)

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

    def __str__(self):
        return "Node %d (type=%s, nInNodes=%d, nOutNodes=%d, dim_in=%d, dim_out=%d)" % (self._id, self._nodeType, self._nInputs, self._nOutputs, self._dim_in, self._dim_out)


class InputNode(Node):

    def __init__(self, nin):
        Node.__init__(self, T_Node_Type.INPUT_NODE)
        self._dim_in = nin
        self._dim_out = nin

    def _setInNodes(self, nodes):
        raise utils.GraphStructureError

    def instantiate(self):
        self._instantiated = True

    def f(self, x):
        return x

    def df(self, x, dEdo=None):
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
            self.w = np.asarray(rng.uniform(low=-4 * np.sqrt(6. / (1 + self._dim_in)), high=4 * np.sqrt(6. / (1 + self._dim_in)), size=(self._dim_in,)), dtype=theano.config.floatX)

        if bias is not None:
            assert bias.shape == (self._dim_out,)
            self.b = bias
        else:
            self.b = np.zeros(shape=(1,), dtype=theano.config.floatX)
        self._instantiated = True

    def f(self, x):
        a = np.dot(self.w, x) + self.b
        o = self.actfunc.f(a)
        self.cache[hash(x.tostring())] = (a, o, None)
        return o

    def df(self, x, dEdo=None):
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
        self._nOutputs = 0
        self._dim_out = nout
        self.W = None
        self.b = None
        self.OutModel = OutputModel(outfunc, errfunc)

    def _setOutNodes(self, nodes):
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
    x = np.asarray([-8., 0.1, 8.])
    w1 = np.asarray([1., -1., 1.])
    w2 = np.asarray([-1., 1., -1.])

    inputnode = InputNode(len(x))
    outputnode = OutputNode(10)

    h1 = HiddenNode()
    h1._setInNodes([inputnode])
    h1._setOutNodes([outputnode])
    h2 = HiddenNode()
    h2._setInNodes([inputnode])
    h2._setOutNodes([outputnode])

    inputnode._setOutNodes([h1, h2])
    outputnode._setInNodes([h1, h2])

    print inputnode
    print h1
    print h2
    print outputnode

    print "Instantiation"
    nodes = [inputnode, h1, h2, outputnode]
    for n in nodes:
        n.instantiate()
        print n