import torch.nn as nn
import torch.nn.functional as F
from lib.statistic import *
from models import *

class IE(nn.Module):

    def __init__(self, model_num, num_classes):
        super(IE, self).__init__()
        self.model_num = model_num
        self.loss = 0
        for model_idx in range(model_num):
            self.add_module('expert-' + str(model_idx), resnet(depth=20, num_classes=num_classes))

    def clean_statistic(self):
        self.oracle = AverageMeter()
        self.top1 = AverageMeter()
        self.experts_top1 = {}
        for name, expert in self.named_children():
            self.experts_top1[name] = AverageMeter()

    def accumulate_statistic(self, size, outputs, target):
        mixture_pred = None
        for name in outputs:
            output = outputs[name]
            pred = F.softmax(output, dim=1)
            mixture_pred = pred if mixture_pred is None else mixture_pred + pred
            self.experts_top1[name].update(correct_count(output, target).data[0], size)
        self.top1.update(correct_count(mixture_pred, target).data[0], size)
        self.oracle.update(oracle_count(outputs, target).data[0], size)

    def print_statistic(self):
        print('Accuracy {top1.avg: .3f}% Oracle Accuracy {oracle.avg: .3f}%'.format(top1=self.top1, oracle=self.oracle))
        for name in self.experts_top1:
            print('{expert} Accuracy {top1.avg:.3f}%'.format(expert=name, top1=self.experts_top1[name]))


    def forward(self, input, target):
        criterion = nn.CrossEntropyLoss().cuda()
        self.loss = 0

        outputs = {}
        for name, expert in self.named_children():
            output = expert(input)
            self.loss += criterion(output, target)
            outputs[name] = output

        self.accumulate_statistic(input.size(0), outputs, target)

    def backward(self):
        return self.loss.backward()







