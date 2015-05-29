__author__ = 'Johannes'

from config import Config
import gzip
import cPickle
import numpy as np
from PIL import Image
from os.path import join
from sklearn.metrics import precision_recall_fscore_support
from movielens import MovielensDataset

def get_full_path(*path):
    return join(Config.ROOT_DIR, *path)


def load_mnist():
    with gzip.open(get_full_path(Config.PATH_DATA_ROOT, Config.MNIST_DATA_FILE), 'rb') as f:
        tr, te, vl = cPickle.load(f)
    return {'train': tr,
            'test': te,
            'valid': vl,
            'name': Config.STR_MNIST,
            'classes': 10}

def load_cifar():
    data = []
    labels = []
    for i in range(1,6):
        fo = open(get_full_path(Config.PATH_DATA_ROOT, Config.CIFAR_DATA_FILE, "data_batch_%d" %i), 'rb')
        dict = cPickle.load(fo)
        data.append(dict["data"])
        labels.append(dict["labels"])
        fo.close()
    data = np.vstack(data).astype(np.float)
    labels = np.hstack(labels)
    data /= data.max()

    fo = open(get_full_path(Config.PATH_DATA_ROOT, Config.CIFAR_DATA_FILE, "test_batch"), 'rb')
    dict = cPickle.load(fo)
    testdata = (dict["data"].astype(np.float) / float(np.max(dict["data"])), np.asarray(dict["labels"]))
    fo.close()
    return {'train': (data, labels),
            'test': testdata,
            'name': Config.STR_CIFAR,
            'classes': 10}

def load_movielens():
    data = MovielensDataset(get_full_path(Config.PATH_DATA_ROOT, Config.MOVIELENS_DATA_FILE))
    uamat = data.getUsersAttributesMatrix()
    uimat = data.getUserItemMatrix()
    return {'dataObj': data,
            'userAttributeMatrix': uamat,
            'userItemMatrix': uimat,
            'name': Config.STR_MOVIELENS}

def code1ofK(labels, K):
    KcodedLabels = []
    for l in labels:
        codedK = np.zeros(shape=(K,))
        codedK[int(l)] = 1.
        KcodedLabels.append(codedK)
    return KcodedLabels

def bool2int(x):
    x = x[::-1]
    y = 0
    for i,j in enumerate(x):
        y += j<<i
    return y


def validate(goldLabels, predictedLabels, K):
    labels = [str(class_label) for class_label in range(K)]
    (pre, rec, f1, sup) = precision_recall_fscore_support(goldLabels, predictedLabels, beta=1.0, labels=labels, pos_label=1, average=None, warn_for=('precision', 'recall', 'f-score'), sample_weight=None)
    np.set_printoptions(precision=3)
    return (pre, rec, f1, sup)


def _scale(x):
    eps = 1e-8
    x = x.copy()
    x -= x.min()
    x *= 1.0 / (x.max() + eps)
    return 255.0*x


def visualize_weights_GREY(weights, panel_shape, tile_size):

    margin_x = np.zeros((tile_size[0], 1)) + 255.
    margin_y = np.zeros((1, (tile_size[1] + 1) * panel_shape[1])) + 255.

    patch_of_rows = []
    for y in range(panel_shape[0]):
        # for each row y : plot all filters at columns x
        row_of_patches = []
        for x in range(panel_shape[1]):
            filterIdx = y * panel_shape[1] + x
            if filterIdx < len(weights):
                row_of_patches.append(np.hstack([_scale(weights[filterIdx].reshape(tile_size)), margin_x]))
            else:
                row_of_patches.append(np.hstack([np.zeros(tile_size), margin_x]))
        rowPatch = np.hstack(row_of_patches)
        patch_of_rows.append(np.vstack([rowPatch, margin_y]))
    image = np.vstack(patch_of_rows)

    img = Image.fromarray(image)
    img = img.convert('RGB')
    return img

def visualize_weights_RGB(weights, panel_shape=(10, 10)):

    tile_len = np.sqrt(weights.shape[1] / 3)
    tile_size = (tile_len, tile_len)
    panel_shape = panel_shape

    margin_x = np.zeros((tile_size[0], 1, 3)) + 255.
    margin_y = np.zeros((1, (tile_size[1] + 1) * panel_shape[1], 3)) + 255.

    patch_of_rows = []
    for y in range(panel_shape[0]):
        # for each row y : plot all filters at columns x
        row_of_patches = []
        for x in range(panel_shape[1]):
            filterIdx = y * panel_shape[1] + x
            if filterIdx < len(weights):
                filter = _scale(np.reshape(np.reshape(weights[filterIdx], (3, 1024)).T, (tile_size[0], tile_size[1], 3)))
                row_of_patches.append(np.hstack([filter, margin_x]))
            else:
                row_of_patches.append(np.hstack([np.zeros((tile_size[0], tile_size[1], 3)), margin_x]))
        rowPatch = np.hstack(row_of_patches)
        patch_of_rows.append(np.vstack([rowPatch, margin_y]))
    image = np.vstack(patch_of_rows).astype(np.uint8)

    img = Image.fromarray(image, mode="RGB")
    return img


def visualize_weights(weights, panel_shape, filter_shape, mode):
    if mode == "RGB":
        return visualize_weights_RGB(weights, panel_shape)
    if mode == "GREY":
        return visualize_weights_GREY(weights, panel_shape, filter_shape)


class GraphStructureError(Exception):
    def __init__(self):
        Exception.__init__(self)
