import torch.nn as nn
import torchvision.transforms as transforms
import math
import nnan_dense
import _ext.nn as enn

__all__ = ['resnet_nan', 'resnet18_nan', 'resnet34_nan', 'resnet50', 'resnet101', 'resnet152']

#snn = nnan.NNaNUnit(dims=[10,10,10])


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def init_model(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        #self.snn = nnan.NNaNUnit(dims=[15,15])
        self.snn = nnan_dense.NNaNUnit(dims=[10,10,10])
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        #out = self.relu(out)
        out = self.snn(out)
        #out = snn(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        #out = self.relu(out)
        out = self.snn(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        #self.relu = nn.ReLU(inplace=True)
        #self.snn = nnan.NNaNUnit(dims=[15,15])
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        #out = self.relu(out)
        #out = self.snn(out)
        #out = snn(out)

        out = self.conv2(out)
        out = self.bn2(out)
        #out = self.relu(out)
        out = self.snn(out)
        #out = snn(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        #out = self.relu(out)
        out = self.snn(out)
        #out = snn(out)

        return out


class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feats(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNet_imagenet(ResNet):

    def __init__(self, num_classes=1000,
                 block=Bottleneck, layers=[3, 4, 23, 3]):
        super(ResNet_imagenet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        #self.snn = nnan.NNaNUnit(dims = [30,30,30])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.feats = nn.Sequential(self.conv1,
                                   self.bn1,
                                   #self.relu,
                                   self.snn,                                 
                                   self.maxpool,

                                   self.layer1,
                                   self.layer2,
                                   self.layer3,
                                   self.layer4,

                                   self.avgpool)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        init_model(self)
        self.regime = {
            0: {'optimizer': 'SGD', 'lr': 1e-1,
                'weight_decay': 1e-4, 'momentum': 0.9},
            30: {'lr': 1e-2},
            60: {'lr': 1e-3},
            90: {'lr': 1e-4}
        }


class ResNet_cifar10(ResNet):

    def __init__(self, num_classes=10,
                 block=BasicBlock, depth=18, conv_init='conv_delta_orthogonal'):
        super(ResNet_cifar10, self).__init__()
        self.inplanes = 16
        self.init_supported = ['conv_delta_orthogonal', 'kaiming_normal']
        if conv_init in self.init_supported:
            self.conv_init = conv_init
        else:
            print('{} is not supported'.format(conv_init))
            self.conv_init = 'kaiming_normal'
        print('initialize conv by {}'.format(conv_init))
        n = int((depth - 2) / 6)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        #self.snn = nnan.NNaNUnit(dims = [10,10,10])
        self.snn = nnan_dense.NNaNUnit(dims = [10,10,10])
        self.maxpool = lambda x: x
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.layer4 = lambda x: x
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)
        self.feats = nn.Sequential(self.conv1,
                                   self.bn1,
                                   #self.relu,
                                   self.snn,
                                   self.layer1,
                                   self.layer2,
                                   self.layer3,
                                   self.avgpool)
        #init_model(self)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.conv_init == self.init_supported[0]:
                    enn.init.conv_delta_orthogonal_(m.weight)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


        self.regime = {
            0: {'optimizer': 'SGD', 'lr':  1e-1,
                'weight_decay': 1e-4, 'momentum': 0.9},
            81: {'lr': 1e-2},
            122: {'lr':  1e-3, 'optimizer': 'SGD'},
            164: {'lr':  1e-4}
        }


def resnet_nan(**kwargs):
    num_classes, depth, dataset = map(
        kwargs.get, ['num_classes', 'depth', 'dataset'])
    if dataset == 'imagenet':
        num_classes = num_classes or 1000
        depth = depth or 18
        if depth == 18:
            return ResNet_imagenet(num_classes=num_classes,
                                   block=BasicBlock, layers=[2, 2, 2, 2])
        if depth == 34:
            return ResNet_imagenet(num_classes=num_classes,
                                   block=BasicBlock, layers=[3, 4, 6, 3])
        if depth == 50:
            return ResNet_imagenet(num_classes=num_classes,
                                   block=Bottleneck, layers=[3, 4, 6, 3])
        if depth == 101:
            return ResNet_imagenet(num_classes=num_classes,
                                   block=Bottleneck, layers=[3, 4, 23, 3])
        if depth == 152:
            return ResNet_imagenet(num_classes=num_classes,
                                   block=Bottleneck, layers=[3, 8, 36, 3])

    elif dataset == 'cifar10':
        num_classes = num_classes or 10
        depth = depth or 44
        return ResNet_cifar10(num_classes=num_classes,
                              block=BasicBlock, depth=depth)
    elif dataset == 'cifar100':
        num_classes = num_classes or 100
        depth = depth or 44
        return ResNet_cifar10(num_classes=num_classes,
                              block=BasicBlock, depth=depth)

def resnet18_nan(**kwargs):
    num_classes, dataset = map(
        kwargs.get, ['num_classes', 'dataset'])
    depth = 18
    if dataset == 'imagenet':
        num_classes = num_classes or 1000
        return ResNet_imagenet(num_classes=num_classes,
                                   block=BasicBlock, layers=[2, 2, 2, 2])
    elif dataset == 'cifar10':
        num_classes = num_classes or 10
        return ResNet_cifar10(num_classes=num_classes,
                              block=BasicBlock, depth=depth)
    elif dataset == 'cifar100':
        num_classes = num_classes or 100
        return ResNet_cifar10(num_classes=num_classes,
                              block=BasicBlock, depth=depth)

def resnet34_nan(**kwargs):
    num_classes, dataset = map(
        kwargs.get, ['num_classes', 'dataset'])
    depth = 34
    if dataset == 'imagenet':
        num_classes = num_classes or 1000
        return ResNet_imagenet(num_classes=num_classes,
                               block=BasicBlock, layers=[3, 4, 6, 3])
    elif dataset == 'cifar10':
        num_classes = num_classes or 10
        return ResNet_cifar10(num_classes=num_classes,
                              block=BasicBlock, depth=depth)
    elif dataset == 'cifar100':
        num_classes = num_classes or 100
        return ResNet_cifar10(num_classes=num_classes,
                              block=BasicBlock, depth=depth)

def resnet50(**kwargs):
    num_classes, dataset = map(
        kwargs.get, ['num_classes', 'dataset'])
    depth = 50
    if dataset == 'imagenet':
        num_classes = num_classes or 1000
        return ResNet_imagenet(num_classes=num_classes,
                               block=Bottleneck, layers=[3, 4, 6, 3])
    elif dataset == 'cifar10':
        num_classes = num_classes or 10
        return ResNet_cifar10(num_classes=num_classes,
                              block=BasicBlock, depth=depth)
    elif dataset == 'cifar100':
        num_classes = num_classes or 100
        return ResNet_cifar10(num_classes=num_classes,
                              block=BasicBlock, depth=depth)

def resnet101(**kwargs):
    num_classes, dataset = map(
        kwargs.get, ['num_classes', 'dataset'])
    depth = 101
    if dataset == 'imagenet':
        num_classes = num_classes or 1000
        return ResNet_imagenet(num_classes=num_classes,
                               block=Bottleneck, layers=[3, 4, 23, 3])
    elif dataset == 'cifar10':
        num_classes = num_classes or 10
        return ResNet_cifar10(num_classes=num_classes,
                              block=BasicBlock, depth=depth)
    elif dataset == 'cifar100':
        num_classes = num_classes or 100
        return ResNet_cifar10(num_classes=num_classes,
                              block=BasicBlock, depth=depth)

def resnet152(**kwargs):
    num_classes, dataset = map(
        kwargs.get, ['num_classes', 'dataset'])
    depth = 152
    if dataset == 'imagenet':
        num_classes = num_classes or 1000
        return ResNet_imagenet(num_classes=num_classes,
                               block=Bottleneck, layers=[3, 8, 36, 3])
    elif dataset == 'cifar10':
        num_classes = num_classes or 10
        return ResNet_cifar10(num_classes=num_classes,
                              block=BasicBlock, depth=depth)
    elif dataset == 'cifar100':
        num_classes = num_classes or 100
        return ResNet_cifar10(num_classes=num_classes,
                              block=BasicBlock, depth=depth)
