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


def get_train_target():
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


## validation from here
def get_validation_images():
    """
    :return: list of [175,175,3] with values 0-255
    """
    VAL_FILENAME = 'Training_Data/92_Image_Set/92images.mat'
    data = read_mat(VAL_FILENAME)
    size = data['visual_stimuli'].shape[1]
    X = [data['visual_stimuli'][0, i][1] for i in range(size)]
    return X


def test_get_validation_images():
    X = get_validation_images()

    assert len(X) == 92
    assert all(x.shape[0] == 175 for x in X)
    assert all(x.shape[1] == 175 for x in X)
    assert all(x.shape[2] == 3 for x in X)
    assert type(X[0]) == np.ndarray


def get_validation_target():
    """
    :return: Returns [92,92,15] for 15 subjects
    """
    VAL_TARGET = 'Training_Data/92_Image_Set/target_fmri.mat'
    data = read_h5(VAL_TARGET)
    return data['IT_RDMs'][:]


def test_get_validation_target():
    y = get_validation_target()

    assert y.shape == (92, 92, 15)
    assert type(y) == np.ndarray

class ValidationDataset(torch.utils.data.Dataset):

    def __init__(self, transform=None):
        self.validation_images = get_validation_images()

        self.len_images = len(self.validation_images)
        self.permutations = tuple(itertools.combinations(
            range(self.len_images), 2))

        self.transform = transform

        self.target = get_validation_target().mean(axis=2).astype(np.float32)

        assert self.target.shape == (self.len_images, self.len_images)

    def __len__(self):
        return len(self.permutations)

    def __getitem__(self, index):
        i = self.permutations[index]
        r1 = self.validation_images[i[0]]
        r2 = self.validation_images[i[1]]
        r3 = self.target[i]

        if self.transform:
            r1 = self.transform(r1)
            r2 = self.transform(r2)

        return r1, r2, r3


def test_validation_dataset():
    dataset = ValidationDataset()

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

    nepochs = 25
    learning_rate = .001
    learning_rate_decay = 0.001
    momentum = .9
    tr_batch_size = 200
    val_batch_size = 50

    # Data
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    tr_dataset = TrainDataset(transform=transform)
    tr_loader = torch.utils.data.DataLoader(tr_dataset, num_workers=4, batch_size=tr_batch_size, shuffle=True)

    val_dataset = ValidationDataset(transform=transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, num_workers=4, batch_size=val_batch_size, shuffle=True)

    criterion = torch.nn.MSELoss()
    
    model = cuda_wrap(model)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)  # change optim here!!
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = 100000
    for e in tqdm(range(nepochs)):
        model.train()
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate/(1+e*learning_rate_decay)
        epoch_loss = 0
        val_epoch_loss = 0
        for batch, (x1, x2, y) in enumerate(tr_loader):

            # print(f"Epoch/Batch: {e}/{batch}")

            x1 = cuda_wrap(x1)
            x2 = cuda_wrap(x2)
            y = cuda_wrap(y)
            out = model(x1, x2)

            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.item()

        epoch_loss/=(batch+1)
        tqdm.write(str(epoch_loss))

        if e%5==4:
            model.eval()
            with torch.no_grad():
                for batch, (x1, x2, y) in enumerate(val_loader):

                    # print(f"Epoch/Batch: {e}/{batch}")

                    x1 = cuda_wrap(x1)
                    x2 = cuda_wrap(x2)
                    y = cuda_wrap(y)
                    out = model(x1, x2)

                    loss = criterion(out, y)
                    val_epoch_loss+=loss.item()

                val_epoch_loss/=(batch+1)
                tqdm.write("ValidationLoss: "+str(val_epoch_loss))
                if val_epoch_loss<=best_val_loss:
                    tqdm.write("Saving model... at Models/ in epoch "+str(e))
                    torch.save(model, open('Models/Siamese_Alex.tm', 'wb')) 


def main():
    train(torch.cuda.is_available())
