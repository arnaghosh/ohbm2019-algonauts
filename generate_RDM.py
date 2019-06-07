from siamese import *
from utils import *
import numpy as np
import scipy.io as sio

def get_test_images():
    """
    :return: list of [300,300,3] with values 0-255
    """
    TEST_FILENAME = 'Test_Data/78images.mat'
    data = read_mat(TEST_FILENAME)
    size = data['visual_stimuli'].shape[1]
    X = [data['visual_stimuli'][0, i][1] for i in range(size)]
    return X


def test_get_test_images():
    X = get_test_images()

    assert len(X) == 78
    assert all(x.shape[0] == 400 for x in X)
    assert all(x.shape[1] == 400 for x in X)
    assert all(x.shape[2] == 3 for x in X)
    assert type(X[0]) == np.ndarray

class TrainDatasetRDM(torch.utils.data.Dataset):

    def __init__(self, transform=None):
        self.train_images = get_train_images()

        self.len_images = len(self.train_images)

        self.transform = transform

    def __len__(self):
        return self.len_images

    def __getitem__(self, index):
        r1 = self.train_images[index]
        
        if self.transform:
            r1 = self.transform(r1)

        return r1

class ValidationDatasetRDM(torch.utils.data.Dataset):

    def __init__(self, transform=None):
        self.validation_images = get_validation_images()

        self.len_images = len(self.validation_images)

        self.transform = transform

    def __len__(self):
        return self.len_images

    def __getitem__(self, index):
        r1 = self.validation_images[index]
        
        if self.transform:
            r1 = self.transform(r1)

        return r1

class TestDatasetRDM(torch.utils.data.Dataset):

    def __init__(self, transform=None):
        self.test_images = get_test_images()

        self.len_images = len(self.test_images)

        self.transform = transform

    def __len__(self):
        return self.len_images

    def __getitem__(self, index):
        r1 = self.test_images[index]
        
        if self.transform:
            r1 = self.transform(r1)

        return r1

model_IT = torch.load('Models/Siamese_Alex_IT.tm')
model_IT = model_IT.cpu()
CNN_IT = model_IT.model

model_EVC = torch.load('Models/Siamese_Alex_EVC.tm')
model_EVC = model_EVC.cpu()
CNN_EVC = model_EVC.model

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

## generate 118 image RDM
dataset = TrainDatasetRDM(transform=transform)
loader = torch.utils.data.DataLoader(dataset, num_workers=1, batch_size=1, shuffle=False)

EVC_activations = []
IT_activations = []
with torch.no_grad():
	for iter,x1 in enumerate(loader):
		feats_list_EVC = CNN_EVC(x1)
		EVC_activations.append(feats_list_EVC[2].flatten().numpy())
		feats_list_IT = CNN_IT(x1)
		IT_activations.append(feats_list_IT[6].flatten().numpy())

EVC_activations = np.array(EVC_activations)
IT_activations = np.array(IT_activations)
print(EVC_activations.shape, IT_activations.shape)
EVC_dis = 1- np.corrcoef(EVC_activations)
IT_dis = 1- np.corrcoef(IT_activations)
sio.savemat('118Image_RDM.mat',{'EVC_RDMs':EVC_dis,'IT_RDMs':IT_dis})


## generate 92 image RDM
dataset = ValidationDatasetRDM(transform=transform)
loader = torch.utils.data.DataLoader(dataset, num_workers=1, batch_size=1, shuffle=False)

EVC_activations = []
IT_activations = []
with torch.no_grad():
	for iter,x1 in enumerate(loader):
		feats_list_EVC = CNN_EVC(x1)
		EVC_activations.append(feats_list_EVC[2].flatten().numpy())
		feats_list_IT = CNN_IT(x1)
		IT_activations.append(feats_list_IT[6].flatten().numpy())

EVC_activations = np.array(EVC_activations)
IT_activations = np.array(IT_activations)
print(EVC_activations.shape, IT_activations.shape)
EVC_dis = 1- np.corrcoef(EVC_activations)
IT_dis = 1- np.corrcoef(IT_activations)
sio.savemat('92Image_RDM.mat',{'EVC_RDMs':EVC_dis,'IT_RDMs':IT_dis})


## generate 78 image RDM
dataset = TestDatasetRDM(transform=transform)
loader = torch.utils.data.DataLoader(dataset, num_workers=1, batch_size=1, shuffle=False)

EVC_activations = []
IT_activations = []
with torch.no_grad():
	for iter,x1 in enumerate(loader):
		feats_list_EVC = CNN_EVC(x1)
		EVC_activations.append(feats_list_EVC[2].flatten().numpy())
		feats_list_IT = CNN_IT(x1)
		IT_activations.append(feats_list_IT[6].flatten().numpy())

EVC_activations = np.array(EVC_activations)
IT_activations = np.array(IT_activations)
print(EVC_activations.shape, IT_activations.shape)
EVC_dis = 1- np.corrcoef(EVC_activations)
IT_dis = 1- np.corrcoef(IT_activations)
sio.savemat('78Image_RDM.mat',{'EVC_RDMs':EVC_dis,'IT_RDMs':IT_dis})