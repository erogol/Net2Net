import unittest
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from net2net import wider, deeper


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)



class Net3D(nn.Module):

    def __init__(self):
        super(Net3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv3d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)


    def forward(self, x):
        x = F.relu(F.max_pool3d(self.conv1(x), 2))
        x = F.relu(F.max_pool3d(self.conv2(x), 2))
        x = x.view(-1, x.size(1)*x.size(2)*x.size(3)*x.size(4))
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)



class TestOperators(unittest.TestCase):


    def _create_net(self):
        return Net()


    def test_wider(self):
        net = self._create_net()
        inp = th.autograd.Variable(th.rand(32, 1, 28, 28))

        net.eval()
        out = net(inp)

        conv1, conv2, _ = wider(net._modules['conv1'],
                                net._modules['conv2'],
                                20,
                                noise=False,
                                random_init=False,
                                weight_norm = False)

        net._modules['conv1'] = conv1
        net._modules['conv2'] = conv2

        conv2, fc1, _ = wider(net._modules['conv2'],
                              net._modules['fc1'],
                              60,
                              noise=False,
                              random_init=False,
                              weight_norm=False)
        net._modules['conv2'] = conv2
        net._modules['fc1'] = fc1

        net.eval()
        nout = net(inp)
        assert th.abs((out - nout).sum().data)[0] < 1e-1
        assert nout.size(0) == 32 and nout.size(1) == 10

        # Testing 3D layers
        net = Net3D()
        inp = th.autograd.Variable(th.rand(32, 1, 16, 28, 28))

        net.eval()
        out = net(inp)

        conv1, conv2, _ = wider(net._modules['conv1'],
                                net._modules['conv2'],
                                20,
                                noise=False,
                                random_init=False,
                                weight_norm=False)

        net._modules['conv1'] = conv1
        net._modules['conv2'] = conv2

        conv2, fc1, _ = wider(net._modules['conv2'],
                              net._modules['fc1'],
                              60,
                              out_size=[1, 4, 4],
                              noise=False,
                              random_init=False,
                              weight_norm=False)
        net._modules['conv2'] = conv2
        net._modules['fc1'] = fc1

        net.eval()
        nout = net(inp)
        assert th.abs((out - nout).sum().data)[0] < 1e-1
        assert nout.size(0) == 32 and nout.size(1) == 10

        # testing noise
        net = self._create_net()
        inp = th.autograd.Variable(th.rand(32, 1, 28, 28))

        net.eval()
        out = net(inp)

        conv1, conv2, _ = wider(net._modules['conv1'],
                                net._modules['conv2'],
                                20,
                                noise=1)

        net._modules['conv1'] = conv1
        net._modules['conv2'] = conv2

        conv2, fc1, _ = wider(net._modules['conv2'],
                              net._modules['fc1'],
                              60,
                              noise=1)
        net._modules['conv2'] = conv2
        net._modules['fc1'] = fc1

        net.eval()
        nout = net(inp)
        assert th.abs((out - nout).sum().data)[0] > 1e-1
        assert nout.size(0) == 32 and nout.size(1) == 10


    def test_deeper(self):
        net = self._create_net()
        inp = th.autograd.Variable(th.rand(32, 1, 28, 28))

        net.eval()
        out = net(inp)

        s = deeper(net._modules['conv1'], nn.ReLU, bnorm_flag=True, weight_norm=False, noise=False)
        net._modules['conv1'] = s

        s2 = deeper(net._modules['conv2'], nn.ReLU, bnorm_flag=True, weight_norm=False, noise=False)
        net._modules['conv2'] = s2

        s3 = deeper(net._modules['fc1'], nn.ReLU, bnorm_flag=True, weight_norm=False, noise=False)
        net._modules['fc1'] = s3

        net.eval()
        nout = net(inp)

        assert th.abs((out - nout).sum().data)[0] < 1e-1

        # test for 3D net
        net = Net3D()
        inp = th.autograd.Variable(th.rand(32, 1, 16, 28, 28))

        net.eval()
        out = net(inp)

        s = deeper(net._modules['conv1'], nn.ReLU, bnorm_flag=False, weight_norm=False, noise=False)
        net._modules['conv1'] = s

        # s2 = deeper(net._modules['conv2'], nn.ReLU, bnorm_flag=False, weight_norm=False, noise=False)
        # net._modules['conv2'] = s2

        net.eval()
        nout = net(inp)

        assert th.abs((out - nout).sum().data)[0] < 1e-1, "New layer changes values by {}".format(th.abs(out - nout).sum().data[0])
