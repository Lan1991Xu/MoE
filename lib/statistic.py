import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = 100.0 * self.sum / self.count

def correct_count(output, target, reduce=True):
    _, pred = torch.topk(output, 1, 1, True, True)

    ans = (pred == target.view(pred.shape)).long()
    if reduce:
        ans = ans.sum()
    return ans


def oracle_count(outputs, target, reduce=True):
    correct_detail = []
    for name in outputs:
        output = outputs[name]
        correct = correct_count(output, target, reduce=False)
        correct_detail.append(correct.view(-1, 1))
    correct_detail = torch.cat(correct_detail, dim=1)
    ans = (correct_detail.sum(dim=1) > 0).long()
    if reduce:
        ans = ans.sum()
    return ans
