import torch
import os

CKPT_ITER = 30

class checkpoint():
    def __init__(self, name):
        #make dir
        if not os.path.exists('result'):
            os.makedirs('result')
        self.fdir = 'result/{}'.format(name)
        if not os.path.exists(self.fdir):
            os.makedirs(self.fdir)

    def save(self, model, opti, epoch, best):
        print('save checkpoint ... epoch {}, fdir {}'.format(epoch, self.fdir))
        addition = ''
        if best:
            addition = '-best'
        elif epoch % CKPT_ITER == 0:
            addition = '-epoch-{}'.format(epoch)
        filepath = os.path.join(self.fdir, 'checkpoint{}.pth'.format(addition))
        state = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'opti': opti.state_dict(),
            'best': best
        }
        torch.save(state, filepath)

    def load(self, resume, model, opti):
        if not os.path.isfile(resume):
            print("=> no checkpoint found at '{}'".format(resume))
            raise RuntimeError
        ckpt = torch.load(resume)
        model.load_state_dict(ckpt['model'])
        opti.load_state_dict(ckpt['opti'])
        print("=> loaded checkpoint '{}' (epoch {})".format(resume, ckpt['epoch']))
        return model, opti
