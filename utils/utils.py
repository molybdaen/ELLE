__author__ = 'Johannes'

import gzip
import cPickle
import numpy
from PIL import Image
import os
from sklearn.metrics import precision_recall_fscore_support

def load_mnist():
    with gzip.open(r"../data/mnist.pkl.gz", 'rb') as f:
        tr,te,vl = cPickle.load(f)
    return tr, te, vl


def validate(goldLabels, predictedLabels):
    (pre, rec, f1, sup) = precision_recall_fscore_support(goldLabels, predictedLabels, beta=1.0, labels=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], pos_label=1, average=None, warn_for=('precision', 'recall', 'f-score'), sample_weight=None)
    numpy.set_printoptions(precision=3)
    print pre
    print rec
    print f1
    print sup
    return (pre, rec, f1, sup)

def visualize_weights(weights, panel_shape, tile_size):

    def scale(x):
        eps = 1e-8
        x = x.copy()
        x -= x.min()
        x *= 1.0 / (x.max() + eps)
        return 255.0*x

    margin_y = numpy.zeros(tile_size[1])
    margin_x = numpy.zeros((tile_size[0] + 1) * panel_shape[0])
    image = margin_x.copy()

    for y in range(panel_shape[1]):
        tmp = numpy.hstack( [ numpy.c_[ scale( x.reshape(tile_size) ), margin_y ] for x in weights[y*panel_shape[0]:(y+1)*panel_shape[0]]])
        tmp = numpy.vstack([tmp, margin_x])
        image = numpy.vstack([image, tmp])

    img = Image.fromarray(image)
    img = img.convert('RGB')
    return img