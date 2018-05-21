import argparse

from mixture_models import *
import torch.optim as optim

from lib.dataset import dataset
from lib.checkpoint import checkpoint
from lib.timer import timer
from lib.solver import solver as solver_

parser = argparse.ArgumentParser(description='pytorch MoE training')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128),only used for train')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('-ds', '--dataset', default='cifar100', type=str, help='dataset (cifar10, cifar100, SVHN)')
parser.add_argument('--model-num', default=2, type=int, help='the number of networks')
parser.add_argument('--method', default='MCL', type=str, help='moe name')
parser.add_argument('--name', default='regular', type=str, help='name')

def main():
    args = parser.parse_args()

    ds = dataset(args.dataset)
    ckpt = checkpoint(args.name)
    solver = solver_()

    #todo: get 'moe' by args.method
    # build model
    moe = IE(args.model_num, ds.num_classes()).cuda()
    optimizer = optim.SGD(moe.parameters(), lr=args.lr, nesterov=True, momentum=args.momentum, weight_decay=args.weight_decay)

    #resume from file
    if args.resume:
        moe, optimizer = ckpt.load(args.resume, moe, optimizer)

    if args.evaluate:
        solver.validate(dataset.testloader(), moe, optimizer)
        #evaluate
        return

    best_prec = 0
    tm = timer('training')

    for epoch in range(args.epochs):
        solver.pre_train(epoch, args.epochs, optimizer, args.lr)
        solver.train(ds.trainloader(args.batch_size), moe, optimizer)
        prec = solver.validate(ds.testloader(), moe, optimizer)

        is_best = prec > best_prec
        best_prec = max(best_prec, prec)

        ckpt.save(moe, optimizer, epoch, is_best)

        tm.log(epoch + 1, args.epochs)

    print('finished. best_prec: {:.4f}'.format(best_prec))

if __name__ == '__main__':
    main()