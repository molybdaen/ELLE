__author__ = 'johannesjurgovsky'


from os.path import dirname


class Config(object):

    ROOT_DIR = dirname(__file__)
    PATH_DATA_ROOT = "data"
    PATH_EVAL_ROOT = "eval"

    MNIST_DATA_FILE = "mnist.pkl.gz"
    CIFAR_DATA_FILE = "cifar-10-batches-py"
    MOVIELENS_DATA_FILE = "ml-m1"
    CONNECTIVITY_FILE = "basic.csv"
