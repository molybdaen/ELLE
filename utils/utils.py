__author__ = 'Johannes'

from config import Config
import gzip
import cPickle
import numpy as np
from PIL import Image
from os.path import join
from sklearn.metrics import precision_recall_fscore_support


def get_full_path(*path):
    return join(Config.ROOT_DIR, *path)


def load_mnist():
    with gzip.open(get_full_path(Config.PATH_DATA_ROOT, Config.MNIST_DATA_FILE), 'rb') as f:
        tr,te,vl = cPickle.load(f)
    return tr, te, vl


def code1ofK(labels, K):
    KcodedLabels = []
    for l in labels:
        codedK = np.zeros(shape=(K,))
        codedK[int(l)] = 1.
        KcodedLabels.append(codedK)
    return KcodedLabels


def validate(goldLabels, predictedLabels):
    (pre, rec, f1, sup) = precision_recall_fscore_support(goldLabels, predictedLabels, beta=1.0, labels=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], pos_label=1, average=None, warn_for=('precision', 'recall', 'f-score'), sample_weight=None)
    np.set_printoptions(precision=3)
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

    margin_x = np.zeros((tile_size[0], 1)) + 255.
    margin_y = np.zeros((1, (tile_size[1] + 1) * panel_shape[1])) + 255.

    patch_of_rows = []
    for y in range(panel_shape[0]):
        # for each row y : plot all filters at columns x
        row_of_patches = []
        for x in range(panel_shape[1]):
            filterIdx = y * panel_shape[1] + x
            if filterIdx < len(weights):
                row_of_patches.append(np.hstack([scale(weights[filterIdx].reshape(tile_size)), margin_x]))
            else:
                row_of_patches.append(np.hstack([np.zeros(tile_size), margin_x]))
        rowPatch = np.hstack(row_of_patches)
        patch_of_rows.append(np.vstack([rowPatch, margin_y]))
    image = np.vstack(patch_of_rows)

    img = Image.fromarray(image)
    img = img.convert('RGB')
    return img


class GraphStructureError(Exception):
    def __init__(self):
        Exception.__init__(self)