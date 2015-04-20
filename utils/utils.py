__author__ = 'Johannes'

import gzip
import cPickle
import numpy
import Image

def load_mnist():
    with gzip.open(r"C:\Users\Johannes\Documents\EEXCESS\useritemmatrix\movielens\ml-1m\mnist.pkl.gz", 'rb') as f:
        tr,te,vl = cPickle.load(f)
    return tr, te, vl

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