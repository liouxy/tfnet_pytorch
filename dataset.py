import torch.utils.data as data

from os import listdir
from os.path import join
from PIL import Image

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ["mul.tif"])

def load_image(filepath):
    img=Image.open(filepath)
    return img

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x.split('_')[0]) for x in listdir(image_dir) if is_image_file(x)]
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input_pan = load_image('%s_pan.tif'%self.image_filenames[index])
        input_lr = load_image('%s_lr.tif'%self.image_filenames[index])
        input_lr_u = load_image('%s_lr_u.tif'%self.image_filenames[index])
        target = load_image('%s_mul.tif'%self.image_filenames[index])
        filename = int(self.image_filenames[index].split('/')[-1])
        if self.input_transform:
            input_pan = self.input_transform(input_pan)
            input_lr = self.input_transform(input_lr)
            input_lr_u = self.input_transform(input_lr_u)
        if self.target_transform:
            target = self.target_transform(target)

        return input_pan, input_lr, input_lr_u, target, filename

    def __len__(self):
        return len(self.image_filenames)

