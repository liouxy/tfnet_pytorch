from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
from torchvision.transforms import Compose, ToTensor
from transforms import Stretch
from dataset import DatasetFromFolder

def input_transform():
    return Compose([ToTensor(), Stretch()])

def target_transform():
    return Compose([ToTensor(), Stretch()])

def get_training_set(root_dir):
    train_dir = join(root_dir, "train")
    return DatasetFromFolder(train_dir,
                             input_transform=input_transform(),
                             target_transform=target_transform())

def get_test_set(root_dir):
    train_dir = join(root_dir, "test")
    return DatasetFromFolder(train_dir,
                             input_transform=input_transform(),
                             target_transform=target_transform())
