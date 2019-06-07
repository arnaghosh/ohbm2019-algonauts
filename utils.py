import h5py
import scipy.io


def read_mat(filename):
    return scipy.io.loadmat(filename)


def read_h5(filename):
    return h5py.File(filename, 'r')
