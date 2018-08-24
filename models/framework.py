# -*- coding: utf-8 -*-
import torch.nn as nn
import torchvision.transforms as transforms
import math
import nnan
import torch
import time
import numpy as np
from graphviz import Digraph 
from torch.autograd import Variable
__all__ = ['resnet_nan', 'resnet18_nan', 'resnet34_nan', 'resnet50', 'resnet101', 'resnet152']

snn = nnan.NNaNUnit(dims=[10,10,10])
#timer = mtime.Timer()

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
        #self.snn = nnan.NNaNUnit(dims=[10,10,10])
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        #out = self.relu(out)
        #out = self.snn(out)
        torch.cuda.synchronize()
        start = time.time()
        out = snn(out)
        torch.cuda.synchronize()
        end = time.time()
        print ('Do once snn need {:.3f}ms ').format((end-start)*1000)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        #out = self.relu(out)
        out = snn(out)

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
        #self.snn = nnan.NNaNUnit(dims=[10,10,10])
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        #out = self.relu(out)
        #out = self.snn(out)
        out = snn(out)

        out = self.conv2(out)
        out = self.bn2(out)
        #out = self.relu(out)
        #out = self.snn(out)
        out = snn(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        #out = self.relu(out)
        #out = self.snn(out)
        out = snn(out)

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
        #self.snn = nnan.NNaNUnit(dims = [10,10,10])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.feats = nn.Sequential(self.conv1,
                                   self.bn1,
                                   #self.relu,
                                   #self.snn,
                                   snn,
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
                 block=BasicBlock, depth=18):
        super(ResNet_cifar10, self).__init__()
        self.inplanes = 16
        n = int((depth - 2) / 6)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        #self.snn = nnan.NNaNUnit(dims = [10,10,10])
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
                                   #timer.tic(),
                                   snn,
                                   #timer.toc(),
                                   #print ('Do once snn need {:.3f}ms ').format(timer.total_time*1000),
                                   self.layer1,
                                   self.layer2,
                                   self.layer3,
                                   self.avgpool)
        init_model(self)

        self.regime = {
            0: {'optimizer': 'SGD', 'lr':  1e-1,
                'weight_decay': 1e-4, 'momentum': 0.9},
            81: {'lr': 1e-2},
            122: {'lr':  1e-3, 'optimizer': 'SGD'},
            164: {'lr':  1e-4}
        }
        
#print ('Do once snn need {:.3f}ms ').format(timer.total_time*1000)

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
               
def make_dot(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}
 
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()
 
    def size_to_str(size):
        return '('+(', ').join(['%d' % v for v in size])+')'
 
    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)
    add_nodes(var.grad_fn)
    return dot

if __name__ == '__main__':  
    net = ResNet_cifar10(num_classes=10,
                              block=BasicBlock, depth=34)  
    xs = np.linspace(-10, 10, 1000)
    x = torch.from_numpy(xs)
    x = Variable(input_var.type(torch.cuda.FloatTensor), volatile=True)
    #x = Variable(torch.randn(1, 1, 1024,1024))  
    y = net(x)  
    g = make_dot(y)  
    g.view()  
  
    params = list(net.parameters())  
    k = 0  
    for i in params:  
        l = 1  
        print("该层的结构：" + str(list(i.size())))  
        for j in i.size():  
            l *= j  
        print("该层参数和：" + str(l))  
        k = k + l  
    print("总参数数量和：" + str(k))
