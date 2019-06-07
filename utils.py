import h5py
import scipy.io


def read_mat(filename):
    return scipy.io.loadmat(filename)


def read_h5(filename):
    return h5py.File(filename, 'r')

def get_train_images():
    """
    :return: list of [300,300,3] with values 0-255
    """
    TRAIN_FILENAME = 'data/algonautsChallenge2019/Training_Data/118_Image_Set/118images.mat'
    data = read_mat(TRAIN_FILENAME)
    size = data['visual_stimuli'].shape[1]
    X = [data['visual_stimuli'][0, i][1] for i in range(size)]
    return X


def test_get_train_images():
    X = get_train_images()

    assert len(X) == 118
    assert all(x.shape[0] == 300 for x in X)
    assert all(x.shape[1] == 300 for x in X)
    assert all(x.shape[2] == 3 for x in X)
    assert type(X[0]) == np.ndarray


def get_train_target() -> np.ndarray:
    """
    :return: Returns [118,118,15] for 15 subjects
    """
    TRAIN_TARGET = 'data/algonautsChallenge2019/Training_Data/118_Image_Set/target_fmri.mat'
    data = read_h5(TRAIN_TARGET)
    return data['IT_RDMs'][:]


def test_get_train_target():
    y = get_train_target()

    assert y.shape == (118, 118, 15)
    assert type(y) == np.ndarray
