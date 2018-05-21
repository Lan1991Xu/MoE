from torch.autograd import Variable

LR_DECAY = 0.1
LR_GAP = 60

class solver():
    def __init__(self):
        # self.model = model
        # self.optimizer = optimizer
        # self.epoch = epoch
        pass

    def pre_train(self, epoch, total_epoch, optimizer, lr):
        factor = 1

        for i in range(total_epoch / LR_GAP):
            if epoch >= LR_GAP * (i + 1):
                factor *= LR_DECAY
        lr *= factor

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        print('training epoch {}/{}, LR = {lr:.4f}'.format(epoch+1, total_epoch, lr=lr))

    def train(self, dataloader, model, optimizer):
        model.train()
        model.clean_statistic()

        for ix, (input, target) in enumerate(dataloader):
            input_var, target_var = Variable(input.cuda()), Variable(target.cuda())
            optimizer.zero_grad()
            model.forward(input_var, target_var) #forward
            model.backward() #backward
            optimizer.step()

        model.print_statistic()

    def validate(self, dataloader, model, optimizer):
        model.eval()
        model.clean_statistic()

        for ix, (input, target) in enumerate(dataloader):
            input_var, target_var = Variable(input.cuda()), Variable(target.cuda())
            model.forward(input_var, target_var)

        model.print_statistic()