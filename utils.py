    
import h5py
import scipy.io
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from tqdm import tqdm
import itertools


def read_mat(filename):
    return scipy.io.loadmat(filename)


def read_h5(filename):
    return h5py.File(filename, 'r')

def get_train_images():
    """
    :return: list of [300,300,3] with values 0-255
    """
    TRAIN_FILENAME = 'Training_Data/118_Image_Set/118images.mat'
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
    TRAIN_TARGET = 'Training_Data/118_Image_Set/target_fmri.mat'
    data = read_h5(TRAIN_TARGET)
    return data['IT_RDMs'][:]


def test_get_train_target():
    y = get_train_target()

    assert y.shape == (118, 118, 15)
    assert type(y) == np.ndarray

class TrainDataset(torch.utils.data.Dataset):

    def __init__(self, transform=None):
        self.train_images = get_train_images()

        self.len_images = len(self.train_images)
        self.permutations = tuple(itertools.combinations(
            range(self.len_images), 2))

        self.transform = transform

        self.target = get_train_target().mean(axis=2).astype(np.float32)

        assert self.target.shape == (self.len_images, self.len_images)

    def __len__(self):
        return len(self.permutations)

    def __getitem__(self, index):
        i = self.permutations[index]
        r1 = self.train_images[i[0]]
        r2 = self.train_images[i[1]]
        r3 = self.target[i]

        if self.transform:
            r1 = self.transform(r1)
            r2 = self.transform(r2)

        return r1, r2, r3


def test_train_dataset():
    dataset = TrainDataset()

    RANDOM = 34

    d = dataset[RANDOM]

    assert len(d) == 3
    assert d[0].shape == d[1].shape == (300, 300, 3)
    assert type(d[2]) == np.float64


def train(model,cuda=False):
    def cuda_wrap(obj):
        if cuda:
            return obj.cuda()
        return obj

    nepochs = 10
    learning_rate = .001
    momentum = .9
    batch_size = 8

    # Data
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = TrainDataset(transform=transform)
    loader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=batch_size, shuffle=True)
    criterion = torch.nn.MSELoss()
    
    model = cuda_wrap(model)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)  # change optim here!!
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for e in tqdm(range(nepochs)):

        for batch, (x1, x2, y) in enumerate(loader):

            # print(f"Epoch/Batch: {e}/{batch}")

            x1 = cuda_wrap(x1)
            x2 = cuda_wrap(x2)
            y = cuda_wrap(y)
            out = model(x1, x2)

            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tqdm.write(str(loss))

    print("Saving model... at Models/")
    torch.save(model, open('Models/Siamese_Alex.tm', 'wb')) 


def main():
    train(torch.cuda.is_available())