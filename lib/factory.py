from mixture_models import *

def mixture_model_factory(method, model_num, num_classes):
    if method == 'IE':
        return IE(model_num, num_classes).cuda()
    print('method: {} not defined!'.format(method))
    raise RuntimeError
