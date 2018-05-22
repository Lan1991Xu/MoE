import torch.nn as nn
import torch.nn.functional as F
from lib.statistic import *
from models import *
import random

OVERLAP = 2
LAM = 0.75

class DCL(nn.Module):

    def __init__(self, model_num, num_classes):
        super(DCL, self).__init__()
        self.model_num = model_num
        self.loss = 0
        for model_idx in range(model_num):
            self.add_module(self.get_expert_name(model_idx), resnet_feature(depth=get_depth(num_classes), num_classes=num_classes))
        for model_idx in range(model_num):
            self.add_module(self.get_gate_name(model_idx), mlp(num_classes=1))


    def clean_statistic(self):
        self.oracle = AverageMeter()
        self.top1 = AverageMeter()
        self.gate_top1 = AverageMeter()
        self.experts_top1 = {}
        for idx in range(self.model_num):
            self.experts_top1[self.get_expert_name(idx)] = AverageMeter()

    def accumulate_statistic(self, size, outputs, confidence, target):
        mixture_pred = None
        # confidence_sqr = confidence * confidence
        # confidence_pred = confidence_sqr / confidence_sqr.sum(dim=1, keepdim=True)
        confidence_pred = confidence

        for idx in outputs:
            output = outputs[idx]
            # pred = F.softmax(output, dim=1) * confidence_pred[:, idx].unsqueeze(1)
            pred = F.softmax(output, dim=1)
            mixture_pred = pred if mixture_pred is None else mixture_pred + pred
            self.experts_top1[self.get_expert_name(idx)].update(correct_count(output, target).data[0], size)
        #todo: gate pred topk
        self.top1.update(correct_count(mixture_pred, target).data[0], size)
        self.oracle.update(oracle_count(outputs, target).data[0], size)

        if random.random() < 1.0 / 4000:
            print('confidence: {}'.format(confidence))

    def print_statistic(self):
        print('Accuracy {top1.avg: .3f}% Oracle Accuracy {oracle.avg: .3f}%'.format(top1=self.top1, oracle=self.oracle))
        for name in self.experts_top1:
            print('{expert} Accuracy {top1.avg:.3f}%'.format(expert=name, top1=self.experts_top1[name]))

    def get_expert_name(self, idx):
        return 'expert-' + str(idx)

    def get_gate_name(self, idx):
        return 'gate-' + str(idx)

    def get_expert(self, idx):
        for name, model in self.named_children():
            if name == self.get_expert_name(idx):
                return model
        raise RuntimeError

    def get_gate(self, idx):
        for name, model in self.named_children():
            if name == self.get_gate_name(idx):
                return model
        raise RuntimeError

    def forward(self, input, target):
        criterion = nn.CrossEntropyLoss(reduce=False).cuda()
        self.loss = 0

        # for name, expert in self.named_children():
        #     output = expert(input)
        #     self.loss += criterion(output, target)
        #     loss_detail.append(criterion(output, target).view(-1, 1))
        #     outputs[name] = output

        outputs = {}

        loss_detail = []
        confidence_detail = []
        entropy_detail = []
        for idx in range(self.model_num):
            feature, output = self.get_expert(idx)(input)
            outputs[idx] = output

            loss_detail.append(criterion(output, target).view(-1, 1))
            entropy_detail.append(-torch.log(F.softmax(output, dim=1) + 1e-9).mean(dim=1).view(-1, 1))

            confidence = self.get_gate(idx)(feature.detach())
            confidence_detail.append(confidence.view(-1, 1))

        loss_detail = torch.cat(loss_detail, dim=1)
        loss_value, min_k_loss_idx = torch.topk(loss_detail, k=OVERLAP, dim=1, largest=False, sorted=True)
        _ , max_loss_idx = torch.topk(loss_detail, k=self.model_num - OVERLAP, dim=1, largest=True, sorted=True)
        min_loss_idx = min_k_loss_idx[:, 0].unsqueeze(1)

        confidence_detail = torch.cat(confidence_detail, dim=1)
        entropy_detail = torch.cat(entropy_detail, dim=1)
        choosed_expert_entropy = torch.gather(entropy_detail, dim=1, index=min_k_loss_idx)

        self.gate_loss = 0
        for idx in range(self.model_num):
            weight = (((max_loss_idx == idx) + (min_loss_idx == idx)).sum(dim=1)>0).float().data
            self.gate_loss += F.binary_cross_entropy(confidence_detail[:, idx], (min_loss_idx == idx).squeeze(dim=1).float(), weight)

        self.expert_mcl_loss = loss_value.mean()
        self.expert_cmcl_loss = LAM * (entropy_detail.sum(dim=1).mean() - choosed_expert_entropy.sum(dim=1).mean())

        self.accumulate_statistic(input.size(0), outputs, confidence_detail, target)


    def backward(self):
        self.loss = self.expert_mcl_loss + self.expert_cmcl_loss + self.gate_loss
        self.loss.backward()


def get_depth(num_classes):
    if num_classes == 10:
        return 20
    elif num_classes == 100:
        return 32
    return 20