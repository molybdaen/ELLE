__author__ = 'johannesjurgovsky'

from utils import utils
from utils.nnmath import *
import numpy as np
from numpy import random as rng
import theano


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

    # def delTier(self, tierIdx):
    #     del self.tiers[tierIdx]
    #
    # def delFromTier(self, tierIdx, nodeIdx):
    #     del self.tiers[tierIdx][nodeIdx]


class Graph(object):

    NODE_CNT = 0

    def __init__(self, inputs, outputs):

        self.inputnode = InputNode(inputs)
        self.outputnode = OutputNode(outputs, outfunc=T_OutFunc_Type.SOFTMAX, errfunc=T_ErrFunc_Type.CROSS_ENTROPY_MULTINOMIAL)
        h = HiddenNode()

        self.gview = GraphView(self.inputnode, self.outputnode)
        self.gview.insertTierAt(1)
        self.gview.addToTier(1, h)
        self.gview.instantiate()

    def propagate(self, x, y):
        pass

    def backpropagate(self, x, y):
        pass

    def __str__(self):
        return str(self.gview)


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

    def _setAsInputs(self, nodes):
        self._inNodes = []
        for n in nodes:
            n._outNodes.append(self)
            self._inNodes.append(n)
            self._upd_inout()
        self._upd_inout()

    def _setAsOutputs(self, nodes):
        self._outNodes = []
        for n in nodes:
            n._inNodes.append(self)
            self._outNodes.append(n)
            n._upd_inout()
        self._upd_inout()

    def _upd_inout(self):
        self._dim_in = 0
        for n in self._inNodes:
            self._dim_in += n._dim_out
        self._nInputs = len(self._inNodes)
        self._nOutputs = len(self._outNodes)

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
    g = Graph(3, 2)
    print g
