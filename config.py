__author__ = 'johannesjurgovsky'


from os.path import dirname


class Config(object):

    ROOT_DIR = dirname(__file__)
    PATH_DATA_ROOT = "data"

    MNIST_DATA_FILE = "mnist.pkl.gz"
