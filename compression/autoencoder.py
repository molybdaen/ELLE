__author__ = 'Johannes'

import numpy as np
import theano as th
from theano import tensor as T
from numpy import random as rng


class AutoEncoder(object):

    def __init__(self, X, hidden_size, activation_function, output_function):

        assert type(X) is np.ndarray
        assert len(X.shape)==2

        self.data = X
        self.X=X
        self.X = th.shared(name='X', value=np.asarray(self.X, dtype=th.config.floatX), borrow=True)

        self.n = X.shape[1]
        self.m = X.shape[0]

        assert type(hidden_size) is int
        assert hidden_size > 0

        self.hidden_size=hidden_size
        initial_W = np.asarray(rng.uniform(low=-4 * np.sqrt(6. / (self.hidden_size + self.n)), high=4 * np.sqrt(6. / (self.hidden_size + self.n)), size=(self.n, self.hidden_size)), dtype=th.config.floatX)
        self.W = th.shared(value=initial_W, name='W', borrow=True)
        self.b1 = th.shared(name='b1', value=np.zeros(shape=(self.hidden_size,), dtype=th.config.floatX),borrow=True)
        self.b2 = th.shared(name='b2', value=np.zeros(shape=(self.n,), dtype=th.config.floatX),borrow=True)
        self.activation_function=activation_function
        self.output_function=output_function

    def mapUserRatings1ofK(self, index):
        userRatings = self.data[index]
        userRatings1ofK = np.zeros((5, self.n), dtype=th.config.floatX)
        for r in userRatings:
            if r > 0:
                userRatings1ofK[r-1] = 1.0
        return userRatings1ofK


    def getCostnGradSingleUser(self, n_epochs=100, mini_batch_size=1, learning_rate=0.1):
        # index = T.lscalar()
        x = T.matrix('x')

        params = [self.W, self.b1, self.b2]
        hidden = self.activation_function(T.dot(x, self.W)+self.b1)

        output = T.dot(hidden,T.transpose(self.W))+self.b2
        # output = self.output_function(output)
        sumOutputs = T.sum(output, axis=0)
        T.addbroadcast(sumOutputs, 0)
        output = output / sumOutputs

        L = 0.5 * T.sqrt(T.sum((x-output)**2.0, axis=0))**2.0
        cost = T.mean(L)# + 0.01 * T.sum(self.W**2.0)

        updates=[]

        gparams = T.grad(cost,params)

        for param, gparam in zip(params, gparams):
            updates.append((param, param-learning_rate*gparam))

        train = th.function(inputs=[x], outputs=[cost], updates=updates)

        import time
        start_time = time.clock()
        cumco = 0.0
        cou = 0
        for epoch in xrange(n_epochs):
            print "Epoch:",epoch
            for row in xrange(self.m):
                c = train(self.mapUserRatings1ofK(row))
                cumco += np.mean(c)
                cou += 1
            print "Cost: %.4f" % (cumco/cou)
            cumco = 0.0
            cou = 0
        end_time = time.clock()
        print "Average time per epoch=", (end_time-start_time)/n_epochs


    def train(self, n_epochs=100, mini_batch_size=1, learning_rate=0.1):
        index = T.lscalar()
        lr = T.fscalar()
        x=T.matrix('x')
        params = [self.W, self.b1, self.b2]
        hidden = self.activation_function(T.dot(x, self.W)+self.b1)
        output = T.dot(hidden,T.transpose(self.W))+self.b2
        output = self.output_function(output)

        # with L.mean: if there are too few bits in hidden layer, very rarely occurring examples get mapped to the same group as the most frequently occurring examples.
        # with L.sum: rarely occurring examples get mapped to a common group, frequently occurring examples still have their own groups
        #Use cross-entropy loss.
        # y = T.nnet.softmax(output)
        # L = T.nnet.binary_crossentropy(output, x)

        # L = -T.sum(x*T.log(output) + (1-x)*T.log(1-output), axis=1)
        L = 0.5 * T.sqrt(T.sum((x-output)**2.0, axis=1))**2.0
        osp = T.mean(output, 1)
        osparsity = T.sum(0.1 * T.log(0.1/osp) + (1.-0.1) * T.log((1.-0.1)/(1.-osp)))
        hsp = T.mean(hidden, 1)
        hsparsity = T.sum(0.5 * T.log(0.5/hsp) + (1.-0.5) * T.log((1.-0.5)/(1.-hsp)))
        # meanActi = T.sqrt(T.sum((T.mean(hidden, axis=0)-0.5)**2.0))
        # bit3penalty = (0.0625 - meanActi[0])**2.0
        # bit2penalty = (0.125 - meanActi[0])**2.0
        # bit1penalty = (0.25 - meanActi[1])**2.0
        # bit0penalty = (0.5 - meanActi[2])**2.0
        cost=L.mean()# + 0.1 * hsparsity# + 0.001 * T.sum(self.W**2.0)# + 0.001 * T.sum(self.W**2.0)# + 0.01 * (bit3penalty + bit2penalty + bit1penalty + bit0penalty)
        # cost=L.mean() + 0.01 * T.sum(self.W**2.0)# + 0.01 * (bit3penalty + bit2penalty + bit1penalty + bit0penalty)
        # cost=L.mean() + 0.01 * (bit2penalty + bit1penalty + bit0penalty)
        # cost = L.mean() + meanActi
        updates=[]

        #Return gradient with respect to W, b1, b2.
        gparams = T.grad(cost,params)

        #Create a list of 2 tuples for updates.
        for param, gparam in zip(params, gparams):
            updates.append((param, param-lr*gparam))

        #Train given a mini-batch of the data.
        train = th.function(inputs=[index, lr], outputs=[cost], updates=updates,
                            givens={x:self.X[index:index+mini_batch_size,:]})


        import time
        start_time = time.clock()
        cumco = 0.0
        cou = 0
        for epoch in xrange(n_epochs):
            print "Epoch:",epoch
            lrnew = learning_rate*0.98
            dat = self.X.get_value()
            np.random.shuffle(dat)
            self.X.set_value(dat)
            for row in xrange(0,self.m, mini_batch_size):
                c = train(row, lrnew)
                cumco += c[0]
                cou += 1
            print "Cost: %.4f" % (cumco/cou)
            cumco = 0.0
            cou = 0
        end_time = time.clock()
        print "Average time per epoch=", (end_time-start_time)/n_epochs

    def get_hidden(self,data):
        x=T.dmatrix('x')
        hidden = self.activation_function(T.dot(x,self.W)+self.b1)
        transformed_data = th.function(inputs=[x], outputs=[hidden])
        return transformed_data(data)

    def get_reconstruction(self, code):
        c=T.dmatrix('c')
        output = T.dot(c,T.transpose(self.W))+self.b2
        output = self.output_function(output)
        # output = T.nnet.softmax(output)
        reconstruction = th.function(inputs=[c], outputs=[output])
        return reconstruction(code)[0]

    def get_reconstruction_error(self, inputx):
        x=T.dmatrix('x')
        hidden = self.activation_function(T.dot(x,self.W)+self.b1)
        output = T.dot(hidden,T.transpose(self.W))+self.b2
        output = self.output_function(output)
        L = 0.5 * T.sqrt(T.sum((x-output)**2.0, axis=1))**2.0
        # Cost = L.mean()
        transformed_data = th.function(inputs=[x], outputs=[hidden, output, L])
        return transformed_data(inputx)

    def get_reconstruction_error_from_code(self, inputx, code):
        x=T.dmatrix('x')
        c=T.dmatrix('c')
        output = T.dot(c,T.transpose(self.W))+self.b2
        output = self.output_function(output)
        L = 0.5 * T.sqrt(T.sum((x-output)**2.0, axis=1))**2.0
        # Cost = L.mean()
        transformed_data = th.function(inputs=[x, c], outputs=[output, L])
        return transformed_data(inputx, code)

    def get_weights(self):
        return [self.W.get_value(), self.b1.get_value(), self.b2.get_value()]

