__author__ = 'Johannes'

from data import MovielensDataset
import numpy as np
from numpy import random as rng
from scipy.spatial.distance import hamming
from sklearn.decomposition import PCA

PATH_DATA_ROOT = r"C:\Users\Johannes\Documents\EEXCESS\useritemmatrix\movielens\ml-1m"
PATH_DATA_EVAL = r"C:\Users\Johannes\Documents\EEXCESS\useritemmatrix\movielens\ml-1m"

xmax = 100
alpha = 3./4.

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def f(x):
    if x < xmax:
        return (x/xmax)**alpha
    else:
        return 1.

def df(x):
    if x < xmax:
        return alpha*((x/xmax)**(alpha-1.))*(1./xmax)
    else:
        return 0.

def movieSimilarity(ma, mb):
    s = len(ma)
    sims = [0]*5
    for i in xrange(s):
        if ma[i] > 0 and mb[i] > 0:
            d = int(abs(ma[i] - mb[i]))
            sims[d] += 1
    ss = sum(sims)
    if ss == 0:
        ss = 1
    acc = 0.
    for k in xrange(len(sims)):
        acc += (1./ss)*sims[k]*np.exp((5-k))
    return acc


class Coder(object):

    def __init__(self, data, userCodeLen=12, movieCodeLen=11):
        self.data = data
        self.U = data.shape[0]
        self.M = data.shape[1]
        self.movieCodeLen = movieCodeLen
        self.userCodeLen = userCodeLen
        self.MovieEmbeddings = self._initMatrix(self.M, self.movieCodeLen)
        self.UserEmbeddings = self._initMatrix(self.U, self.userCodeLen)
        self.sampleSize = 2
        self.distinctRatings = 6

    def _initMatrix(self, rows, cols):
        """
        embedding matrix with random embedding vectors scaled to unit length
        Uniformly distributed points on unit sphere
        :param rows: Number of instance
        :param cols: Number of dimensions (length of code)
        :return:
        """
        m = rng.normal(0.0, 1.0, (rows, cols))
        magnitudes = np.sqrt(np.sum(m**2., axis=1))[:, np.newaxis]
        return m / magnitudes

    def _getUserContext(self, userIdx):
        simMovies = {}
        for rat in xrange(6):
            simMovies[rat] = self.MovieEmbeddings[[i for i,r in enumerate(self.data[userIdx]) if r == rat]]
        return simMovies

    def _getMovieContext(self, movieIdx):
        simUsers = {}
        for rat in xrange(6):
            simUsers[rat] = self.UserEmbeddings[[i for i,r in enumerate(self.data[:, movieIdx]) if r == rat]]
        return simUsers

    def _getRandomMovieContext(self):
        ma = self.data[:,rng.randint(0,self.M)]
        mb = self.data[:,rng.randint(0,self.M)]
        rDist = {}
        for i in xrange(self.U):
            if ma[i] > 0 and mb[i] > 0:
                d = int(abs(ma[i]-mb[i]))
                if d in rDist:
                    rDist[d].append(i)
                else:
                    rDist[d] = [i]
        return rDist

    def _getUserRatingContext(self, userIdx, movieIdx):
        r = self.data[userIdx, movieIdx]
        if r > 0:
            m = self.data[:,movieIdx]
            u = self.data[userIdx,:]
            simMovies = {}
            for i in xrange(self.M):
                if u[i] > 0:
                    d = int(abs(u[i]-r))
                    if d in simMovies:
                        simMovies[d].append(i)
                    else:
                        simMovies[d] = [i]
            simUsers = {}
            for i in xrange(self.U):
                if m[i] > 0:
                    d = int(abs(m[i]-r))
                    if d in simUsers:
                        simUsers[d].append(i)
                    else:
                        simUsers[d] = [i]
            return (simMovies, simUsers)
        else:
            return None

    def _buildMovieCoOccurrence(self, mat):
        movieCooc = np.zeros(shape=(self.M, self.M))
        for mi in xrange(self.M):
            for mj in xrange(mi,self.M):
                sim = movieSimilarity(mat[:,mi], mat[:,mj])
                movieCooc[mi, mj] = sim
                movieCooc[mj, mi] = sim
            print "row %d" % mi
            print sum(movieCooc[mi,:])
        return movieCooc


    def _getUserMovieContext(self, userIdx, movieIdx):
        return self.UserEmbeddings[[i for i, r in enumerate(self.data[:,movieIdx]) if r == self.data[userIdx][movieIdx]]]

    def _getMoviesWithRating(self, r):
        pass


    def _getCost(self, wi, wj, bi, bj, t):
        return f(t)*(np.dot(wi,wj) + bi + bj - np.log(t))**2.0

    def _getGradient(self, wi, wj, bi, bj, t):
        dC = f(t) * 2.0 * (np.dot(wi,wj) + bi + bj - np.log(t))
        gwi = dC * wj
        gwj = dC * wi
        return (gwi, gwj)

    def train(self):
        pass

    def predict(self):
        pass



if __name__ == '__main__':
    data = MovielensDataset(PATH_DATA_ROOT)
    uamat = data.getUserItemMatrix()
    print np.shape(uamat)

    c = Coder(uamat, userCodeLen=12, movieCodeLen=11)
    print c.MovieEmbeddings.shape
    print "User Context"
    simMovies = c._getUserContext(3)
    for m in simMovies:
        print "Rated with %d: %s" % (m, str(simMovies[m].shape))

    print "Movie Context"
    simUsers = c._getMovieContext(3)
    for k in simUsers:
        print "Rated with %d: %s" % (k, str(simUsers[k].shape))
    print c._getUserMovieContext(10, 3).shape

    a = c._initMatrix(4,3)
    print np.dot(a, a.T)

    cooc = c._buildMovieCoOccurrence(uamat)
    print cooc

    # for i in xrange(3000):
    #     result = c._getUserRatingContext(3, i)
    #     if result is not None:
    #         (simmovies, simusers) = result
    #         print "similarly rated movies:"
    #         print simmovies
    #         print "similarly rating users:"
    #         print simusers