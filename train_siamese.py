from utils import *
from siamese import *
model = SiameseModel(6)
train(model,cuda=True)