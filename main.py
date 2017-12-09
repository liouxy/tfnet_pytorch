import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import Net, TFNet
from data import get_training_set, get_test_set
import random
import os
import time

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--batchSize', type=int, default=125, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=600, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--dataset', type=str, default='/data/pansharpening/QB/TIF')
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--step", type=int, default=250, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=500")
parser.add_argument("--net", type=str, choices={'resnet','tfnet'})
parser.add_argument("--log", type=str, default="log/")
opt = parser.parse_args()

def main():
    print(opt)

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    opt.seed = random.randint(1,10000)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print('===> Loading datasets')
    train_set = get_training_set(opt.dataset)
    test_set = get_test_set(opt.dataset)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    test_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False)

    print("===> Building model")
    if (opt.net=='resnet'):
        model = Net()
    else:
        model = TFNet()
    criterion = nn.L1Loss()

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()


    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    print("===> Training")
    t = time.strftime("%Y%m%d%H%M")
    train_log = open(os.path.join(opt.log, "%s_%s_train.log")%(opt.net, t), "w")
    test_log = open(os.path.join(opt.log, "%s_%s_test.log")%(opt.net, t), "w")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(training_data_loader, optimizer, model, criterion, epoch, train_log)
        if epoch%10==0:
            test(test_data_loader, model, criterion, epoch, test_log)
            save_checkpoint(model, epoch, t)
    train_log.close()
    test_log.close()

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.5 ** (epoch // opt.step))
    return lr

def train(training_data_loader, optimizer, model, criterion, epoch, train_log):
    lr = adjust_learning_rate(optimizer, epoch - 1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print ("epoch =", epoch, "lr =", optimizer.param_groups[0]["lr"])
    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):
        input_pan, input_lr, input_lr_u, target = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]),Variable(batch[3], requires_grad=False)
        if opt.cuda:
            input_pan = input_pan.cuda()
            input_lr = input_lr.cuda()
            input_lr_u = input_lr_u.cuda()
            target = target.cuda()
        if(opt.net=="resnet"):
            output = model(input_pan, input_lr)
        else:
            output = model(input_pan, input_lr_u)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_log.write("{} {:.10f}\n".format(epoch*len(training_data_loader)+iteration, loss.data[0]))
        if iteration%10 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(training_data_loader),
                                                                loss.data[0]))


def test(test_data_loader, model, criterion, epoch, test_log):
    avg_l1 = 0
    for batch in test_data_loader:
        input_pan, input_lr, input_lr_u, target = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]),Variable(batch[3], requires_grad=False)
        if opt.cuda:
            input_pan = input_pan.cuda()
            input_lr = input_lr.cuda()
            input_lr_u = input_lr_u.cuda()
            target = target.cuda()
        if (opt.net == "resnet"):
            output = model(input_pan, input_lr)
        else:
            output = model(input_pan, input_lr_u)
        loss = criterion(output, target)
        avg_l1 += loss
    test_log.write("{} {:.10f}\n".format(epoch, avg_l1 / len(test_data_loader)))
    print("===>Epoch{} Avg. L1: {:.4f} dB".format(epoch, avg_l1 / len(test_data_loader)))

def save_checkpoint(model, epoch, t):
    model_out_path = "model/model/{}_{}/model_epoch_{}.pth".format(opt.net,t,epoch)
    state = {"epoch": epoch, "model": model}

    if not os.path.exists("model/model/{}".format(opt.net)):
        os.makedirs("model/model/{}".format(opt.net))

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()
