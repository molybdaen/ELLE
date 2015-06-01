__author__ = 'johannesjurgovsky'


import theano
import numpy as np
from utils import utils
import gensim


class LLEW2V(object):

    def __init__(self, nvis, vocabularySize, embedSize):
        self.nvis = nvis
        self.vocabularySize = vocabularySize
        self.embedSize = embedSize
        self.k = 10
        self.embeddings = 1./self.embedSize * (np.random.random((self.embedSize, self.vocabularySize)) - 0.5)

    def _normalize(self, data):
        centered = data - np.mean(data)
        normalized = centered / np.linalg.norm(centered, axis=1)[:,None]
        return normalized

    def _get_k_nearest(self, x, data):
        nearestIndices = np.argsort(np.dot(data, x))[::-1]
        return data[nearestIndices[:self.k]]

    def train(self, data):

        for x in data:
            neighbourhood = self._get_k_nearest()
            p = np.sum(neighbourhood, axis=0)



if __name__ == "__main__":
    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    data = utils.load_mnist_rotated()
    train = data['train'][0]
    labels = data['train'][1]
    print train.shape
    print labels.shape
    print data['test'][0].shape
    print data['test'][1].shape
    lw = LLEW2V(100, 3)