__author__ = 'johannesjurgovsky'


from os.path import dirname


class Config(object):

    ROOT_DIR = dirname(__file__)
    PATH_DATA_ROOT = "data"
    PATH_EVAL_ROOT = "eval"

    MNIST_DATA_FILE = "mnist.pkl.gz"
    MNIST_ROTATED_DATA_FILE = "mnist-rotation-new"
    MNIST_ROTATED_FILE = r"mnist-rotated.pkl.gz"
    CIFAR_DATA_FILE = "cifar-10-batches-py"
    MOVIELENS_DATA_FILE = "ml-1m"
    CONNECTIVITY_FILE = "basic.csv"

    STR_MNIST = "MNIST"
    STR_MNIST_ROTATED = "MNIST_ROTATED"
    STR_CIFAR = "CIFAR"
    STR_MOVIELENS = "movielens"
