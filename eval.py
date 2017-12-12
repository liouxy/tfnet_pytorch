import numpy as np

import argparse
import torch
from data import get_test_set
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os

from PIL import Image

parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--dataset', type=str, default='/data/pansharpening/QB/TIF')
parser.add_argument('--checkpoint', type=str)
parser.add_argument("--net", type=str, choices={'resnet','tfnet'})
parser.add_argument('--testBatchSize', type=int, default=96, help='testing batch size')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
opt = parser.parse_args()

test_set = get_test_set(opt.dataset)
test_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)
model = torch.load('model/%s'%opt.checkpoint)

image_path = 'images/%s'%opt.checkpoint

def test(test_data_loader, model):
    model.eval()
    for index,batch in enumerate(test_data_loader):
        input_pan, input_lr, input_lr_u, filename = Variable(batch[0],volatile=True), Variable(batch[1],volatile=True), Variable(batch[2],volatile=True),Variable(batch[4], requires_grad=False,volatile=True)
        if opt.cuda:
            input_pan = input_pan.cuda()
            input_lr_u = input_lr_u.cuda()
        output = model(input_pan, input_lr_u)
        output = output.cpu()
        for i in range(output.data.shape[0]):
            image = (output.data[i]+1)/2.0
            image = image.mul(255).byte()
            image = np.transpose(image.numpy(), (1, 2, 0))
            print (image.shape)
            image = Image.fromarray(image)
            image.save(os.path.join(image_path,'%d_out_tf.tif'%(filename.data[i])))


if not os.path.exists(image_path):
    os.makedirs(image_path)


test(test_data_loader, model['model'])











