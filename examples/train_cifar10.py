from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import sys
sys.path.append('../')
from net2net import wider, deeper
import copy
import time
import numpy as np


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before logging status')
parser.add_argument('--noise', type=float, default=0.01,
                    help='noise variance for wider operator')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

transform = transforms.Compose(
             [transforms.ToTensor(),
              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=True, download=True, transform=transform),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=False, transform=transform),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.pool1 = nn.MaxPool2d(3, 2)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool2 = nn.MaxPool2d(3, 2)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool3 = nn.AvgPool2d(5, 1)
        self.fc1 = nn.Linear(32 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                m.bias.data.fill_(0.0)
            if isinstance(m, nn.Linear):
                m.bias.data.fill_(0.0)

    def forward(self, x):
        try:
            x = self.pool1(F.relu(self.conv1(x)))
            x = self.pool2(F.relu(self.conv2(x)))
            #print(x.size())
            x = self.pool3(F.relu(self.conv3(x)))
            #print(x.size())
            x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.log_softmax(x)
        except RuntimeError:
            print(x.size())

    def net2net_wider(self):
        self.conv1, self.conv2, _ = wider(self.conv1, self.conv2, 12, noise_var=args.noise)
        self.conv2, self.conv3, _ = wider(self.conv2, self.conv3, 24, noise_var=args.noise)
        self.conv3, self.fc1, _ = wider(self.conv3, self.fc1, 48, noise_var=args.noise)
        print(self)

    def net2net_deeper(self):
        s = deeper(self.conv1, nn.ReLU, bnorm_flag=False)
        self.conv1 = s
        s = deeper(self.conv2, nn.ReLU, bnorm_flag=False)
        self.conv2 = s
        s = deeper(self.conv3, nn.ReLU, bnorm_flag=False)
        self.conv3 = s
        print(self)

    def define_wider(self):
        self.conv1 = nn.Conv2d(3, 12, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(12, 24, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(24, 48, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(48*3*3, 256)

    def define_wider_deeper(self):
        self.conv1 = nn.Sequential(nn.Conv2d(3, 12, kernel_size=3, padding=1),
                                   nn.ReLU(),
                                   nn.Conv2d(12, 12, kernel_size=3, padding=1))
        self.conv2 = nn.Sequential(nn.Conv2d(12, 24, kernel_size=3, padding=1),
                                   nn.ReLU(),
                                   nn.Conv2d(24, 24, kernel_size=3, padding=1))
        self.conv3 = nn.Sequential(nn.Conv2d(24, 48, kernel_size=3, padding=1),
                                   nn.ReLU(),
                                   nn.Conv2d(48, 48, kernel_size=3, padding=1))
        self.fc1 = nn.Linear(48*3*3, 256)
        print(self)


model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset), test_loss


def net2net_deeper_recursive(model):
    for name, module in model._modules.items():
        if isinstance(module, nn.Conv2d):
            s = deeper(module, nn.ReLU, bnorm_flag=False)
            model._modules[name] = s
        elif isinstance(module, nn.Sequential):
            module = net2net_deeper_recursive(module)
            model._modules[name] = module
    return model


print("\n\n > Teacher training ... ")
# treacher training
teacher_losses = []
teacher_accus = []
for epoch in range(1, args.epochs + 1):
    train(epoch)
    accu, loss = test()
    teacher_losses.append(loss)
    teacher_accus.append(accu)


# wider student training
print("\n\n > Wider Student training ... ")
model_ = Net()
model_ = copy.deepcopy(model)

del model
model = model_
model.net2net_wider()
model.cuda()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
wider_accus = []
wider_losses = []
for epoch in range(1, args.epochs + 1):
    train(epoch)
    accu, loss = test()
    wider_accus.append(accu)
    wider_losses.append(loss)

#accu_iter = []
#loss_iter = []
#for i in range(10):
#    print("Deeper {} -----".format(i))
#    model_ = Net()
#    model_ = copy.deepcopy(model)
#    model = model_
#    model = net2net_deeper_recursive(model)
#    model.cuda()
#
#    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
#    deeper_accus = []
#    deeper_losses = []
#    for epoch in range(1, args.epochs + 1):
#        train(epoch)
#        accu, loss = test()
#        deeper_accus.append(accu)
#        deeper_losses.append(loss)
#    accu_iter.append(deeper_accus)
#    loss_iter.append(deeper_losses)


# wider + deeper student training
print("\n\n > Wider+Deeper Student training ... ")
model_ = Net()
model_.net2net_wider()
model_ = copy.deepcopy(model)

del model
model = model_
model.net2net_deeper()
model.cuda()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
deeper_accus = []
deeper_losses = []
for epoch in range(1, args.epochs + 1):
    train(epoch)
    accu, loss = test()
    deeper_accus.append(accu)
    deeper_losses.append(loss)


# wider teacher training
print("\n\n > Wider teacher training ... ")
model_ = Net()

del model
model = model_
model.define_wider()
model.cuda()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
wider_teacher_losses = []
wider_teacher_accus = []
for epoch in range(1, 2*args.epochs+1):
    train(epoch)
    accu, loss = test()
    wider_teacher_losses.append(loss)
    wider_teacher_accus.append(accu)



# wider deeper teacher training
print("\n\n > Wider+Deeper teacher training ... ")
model_ = Net()

del model
model = model_
model.define_wider_deeper()
model.cuda()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
wider_deeper_teacher_accus = []
wider_deeper_teacher_losses = []
for epoch in range(1, 3*args.epochs+1):
    train(epoch)
    accu, loss = test()
    wider_deeper_teacher_accus.append(accu)
    wider_deeper_teacher_losses.append(loss)


print(" -> Teacher:\t{}\t{}".format(np.max(teacher_accus),
                                    np.min(teacher_losses)))
print(" -> Wider model:\t{}\t{}".format(np.max(wider_accus),
                                        np.min(wider_losses)))
print(" -> Deeper-Wider model:\t{}\t{}".format(np.max(deeper_accus),
                                               np.min(deeper_losses)))
print(" -> Wider teacher:\t{}\t{}".format(np.max(wider_teacher_accus),
                                          np.min(wider_teacher_losses)))
print(" -> Deeper-Wider teacher:\t{}\t{}".format(np.max(wider_deeper_teacher_accus),
                                                 np.min(wider_deeper_teacher_losses)))
