import pdb
import time
import torch
from torch.autograd import Variable
import models
import argparse

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch measure time of net')

parser.add_argument('--model', '-a', metavar='MODEL', default='resnet',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: alexnet)')



model = models.__dict__[args.model]

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

#GPUID = 1
#resnet34 = model()
#resnet34.cuda(GPUID)

x = torch.rand(1,3,400,400)
#x = Variable(x.cuda(GPUID))

# preheat
y = model(x)
timer = Timer()
timer.tic()
for i in xrange(100):
  y = model(x)
timer.toc()

print ('Do once forward need {:.3f}ms ').format(timer.total_time*1000/100.0)
